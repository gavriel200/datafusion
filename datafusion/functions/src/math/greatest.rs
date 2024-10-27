// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{make_comparator, Array, ArrayRef, BooleanArray, BooleanBuilder};
use arrow::compute::kernels::zip::zip;
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use datafusion_common::{exec_err, plan_err, Result, ScalarValue};
use datafusion_expr::type_coercion::functions::can_coerce_from;
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};
use std::any::Any;

const SORT_OPTIONS: SortOptions = SortOptions {
    descending: false,
    nulls_first: true,
};

#[derive(Debug)]
pub struct GreatestFunc {
    signature: Signature,
}

impl Default for GreatestFunc {
    fn default() -> Self {
        GreatestFunc::new()
    }
}

impl GreatestFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

fn get_larger(lhs: &dyn Array, rhs: &dyn Array) -> Result<BooleanArray> {
    let cmp = make_comparator(lhs, rhs, SORT_OPTIONS)?;

    let len = lhs.len().min(rhs.len());

    let mut builder = BooleanBuilder::with_capacity(len);

    for i in 0..len {
        let ordering = cmp(i, i);
        // Use `is_ge` since we consider nulls smaller than any value
        let is_larger = ordering.is_ge();
        builder.append_value(is_larger);
    }

    Ok(builder.finish())
}

fn keep_larger(lhs: ArrayRef, rhs: ArrayRef) -> Result<ArrayRef> {
    // True for values that we should keep from the left array
    let keep_lhs = get_larger(lhs.as_ref(), rhs.as_ref())?;

    let larger = zip(&keep_lhs, &lhs, &rhs)?;

    Ok(larger)
}

fn keep_larger_scalar(lhs: &ScalarValue, rhs: &ScalarValue) -> Result<ScalarValue> {
    // Handle nulls: consider null as the smallest value
    if lhs.is_null() {
        return Ok(rhs.clone());
    }
    if rhs.is_null() {
        return Ok(lhs.clone());
    }

    if !lhs.data_type().is_nested() {
        return if lhs >= rhs {
            Ok(lhs.clone())
        } else {
            Ok(rhs.clone())
        };
    }

    // If complex type, compare using arrays
    let cmp = make_comparator(
        lhs.to_array()?.as_ref(),
        rhs.to_array()?.as_ref(),
        SORT_OPTIONS,
    )?;

    if cmp(0, 0).is_ge() {
        Ok(lhs.clone())
    } else {
        Ok(rhs.clone())
    }
}

fn find_coerced_type(data_types: &[DataType]) -> Result<DataType> {
    let non_null_types = data_types
        .iter()
        .filter(|t| !matches!(t, DataType::Null))
        .collect::<Vec<_>>();

    if non_null_types.is_empty() {
        return Ok(DataType::Null);
    }

    // Try to find a common data type that all types can be coerced into
    for data_type in &non_null_types {
        let can_coerce_to_all =
            non_null_types.iter().all(|t| can_coerce_from(data_type, t));

        if can_coerce_to_all {
            return Ok((*data_type).clone());
        }
    }

    plan_err!("Cannot find a common type for arguments")
}

impl ScalarUDFImpl for GreatestFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "greatest"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        find_coerced_type(arg_types)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        if args.len() < 2 {
            return exec_err!(
                "greatest was called with {} arguments. It requires at least 2.",
                args.len()
            );
        }

        // Split into scalars and arrays for optimization
        let (scalars, arrays): (Vec<_>, Vec<_>) = args
            .iter()
            .partition(|x| matches!(x, ColumnarValue::Scalar(_)));

        let mut arrays_iter = arrays.iter().filter_map(|x| match x {
            ColumnarValue::Array(a) => Some(a),
            _ => None,
        });

        let first_array = arrays_iter.next();

        let mut largest: ArrayRef;

        // Merge all scalars into one scalar
        let merged_scalar = if !scalars.is_empty() {
            let mut scalars_iter = scalars.iter().map(|x| match x {
                ColumnarValue::Scalar(s) => s.clone(),
                _ => unreachable!(),
            });

            // Initialize with the first scalar
            let mut largest_scalar = scalars_iter.next().unwrap();

            for scalar in scalars_iter {
                largest_scalar = keep_larger_scalar(&largest_scalar, &scalar)?;
            }

            Some(largest_scalar)
        } else {
            None
        };

        // If we only have scalars, return the largest one
        if arrays.is_empty() {
            return Ok(ColumnarValue::Scalar(merged_scalar.unwrap()));
        }

        // We have at least one array
        let first_array = first_array.unwrap();

        if let Some(scalar) = merged_scalar {
            // Start with the scalar and the first array
            largest = keep_larger(
                first_array.clone(),
                scalar.to_array_of_size(first_array.len())?,
            )?;
        } else {
            // Start with the first array
            largest = first_array.clone();
        }

        // Iterate through the remaining arrays
        for array in arrays_iter {
            largest = keep_larger(largest, array.clone())?;
        }

        Ok(ColumnarValue::Array(largest))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() < 2 {
            return exec_err!(
                "greatest was called with {} arguments. It requires at least 2.",
                arg_types.len()
            );
        }

        let coerced_type = find_coerced_type(arg_types)?;

        Ok(vec![coerced_type; arg_types.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date32Array, Float64Array, Int32Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn test_greatest_int32_arrays() {
        let func = GreatestFunc::new();

        let arr1 = ColumnarValue::Array(Arc::new(Int32Array::from(vec![1, 8, 3, 5])));
        let arr2 = ColumnarValue::Array(Arc::new(Int32Array::from(vec![4, 5, 6, 7])));

        let result = func.invoke(&[arr1, arr2]).unwrap();
        let result_array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected an array"),
        };

        let expected = Int32Array::from(vec![4, 8, 6, 7]);
        let result_as_int32 = result_array.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result_as_int32, &expected);
    }

    #[test]
    fn test_greatest_with_nulls() {
        let func = GreatestFunc::new();

        let arr1 = ColumnarValue::Array(Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            Some(5),
        ])));
        let arr2 = ColumnarValue::Array(Arc::new(Int32Array::from(vec![
            Some(4),
            Some(5),
            None,
            Some(7),
        ])));

        let result = func.invoke(&[arr1, arr2]).unwrap();
        let result_array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected an array"),
        };

        let expected = Int32Array::from(vec![Some(4), Some(5), Some(3), Some(7)]);
        let result_as_int32 = result_array.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result_as_int32, &expected);
    }

    #[test]
    fn test_greatest_with_scalars() {
        let func = GreatestFunc::new();

        let arr1 = ColumnarValue::Array(Arc::new(Int32Array::from(vec![
            Some(1),
            Some(8),
            Some(3),
            None,
        ])));
        let scalar = ColumnarValue::Scalar(ScalarValue::Int32(Some(5)));

        let result = func.invoke(&[arr1, scalar]).unwrap();
        let result_array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected an array"),
        };

        let expected = Int32Array::from(vec![Some(5), Some(8), Some(5), Some(5)]);
        let result_as_int32 = result_array.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result_as_int32, &expected);
    }

    #[test]
    fn test_greatest_float_arrays() {
        let func = GreatestFunc::new();

        let arr1 =
            ColumnarValue::Array(Arc::new(Float64Array::from(vec![1.0, 8.0, 3.0, 5.0])));
        let arr2 =
            ColumnarValue::Array(Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0, 7.0])));

        let result = func.invoke(&[arr1, arr2]).unwrap();
        let result_array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected an array"),
        };

        let expected = Float64Array::from(vec![4.0, 8.0, 6.0, 7.0]);
        let result_as_float64 = result_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(result_as_float64, &expected);
    }

    #[test]
    fn test_greatest_string_arrays() {
        let func = GreatestFunc::new();

        let arr1 = ColumnarValue::Array(Arc::new(StringArray::from(vec![
            Some("apple"),
            Some("zebra"),
            Some("ABC"),
            None,
        ])));
        let arr2 = ColumnarValue::Array(Arc::new(StringArray::from(vec![
            Some("banana"),
            Some("yellow"),
            Some("abc"),
            Some("banana"),
        ])));
        let arr3 = ColumnarValue::Array(Arc::new(StringArray::from(vec![
            Some("cherry"),
            Some("xylophone"),
            Some("AbC"),
            None,
        ])));

        let result = func.invoke(&[arr1, arr2, arr3]).unwrap();
        let result_array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected an array"),
        };

        let expected = StringArray::from(vec![
            Some("cherry"),
            Some("zebra"),
            Some("abc"),
            Some("banana"),
        ]);
        let result_as_string =
            result_array.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(result_as_string, &expected);
    }

    #[test]
    fn test_greatest_mixed_types_error() {
        let func = GreatestFunc::new();

        let arr1 = ColumnarValue::Array(Arc::new(Int32Array::from(vec![1, 8, 3, 5])));
        let arr2 = ColumnarValue::Array(Arc::new(StringArray::from(vec![
            Some("apple"),
            Some("banana"),
            Some("cherry"),
            Some("date"),
        ])));

        // Attempt to invoke with mixed types (should result in an error)
        let result = func.invoke(&[arr1, arr2]);

        assert!(result.is_err());
    }

    #[test]
    fn test_greatest_date_arrays() {
        let func = GreatestFunc::new();

        use chrono::NaiveDate;

        let dates1 = vec![
            Some(NaiveDate::from_ymd_opt(2024, 1, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 6, 1)),
            None,
        ];
        let dates2 = vec![
            Some(NaiveDate::from_ymd_opt(2024, 2, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 5, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 1, 1)),
        ];
        let dates3 = vec![
            Some(NaiveDate::from_ymd_opt(2024, 3, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 4, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 2, 1)),
        ];

        let date_to_days = |date_opt: Option<NaiveDate>| {
            date_opt.map(|date| {
                date.signed_duration_since(NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
                    .num_days() as i32
            })
        };

        let arr1_dates: Vec<Option<i32>> = dates1
            .into_iter()
            .map(|arg0: Option<Option<NaiveDate>>| date_to_days(arg0.flatten()))
            .collect();
        let arr2_dates: Vec<Option<i32>> = dates2
            .into_iter()
            .map(|arg0: Option<Option<NaiveDate>>| date_to_days(arg0.flatten()))
            .collect();
        let arr3_dates: Vec<Option<i32>> = dates3
            .into_iter()
            .map(|arg0: Option<Option<NaiveDate>>| date_to_days(arg0.flatten()))
            .collect();

        let arr1 = ColumnarValue::Array(Arc::new(Date32Array::from(arr1_dates)));
        let arr2 = ColumnarValue::Array(Arc::new(Date32Array::from(arr2_dates)));
        let arr3 = ColumnarValue::Array(Arc::new(Date32Array::from(arr3_dates)));

        let result = func.invoke(&[arr1, arr2, arr3]).unwrap();
        let result_array = match result {
            ColumnarValue::Array(array) => array,
            _ => panic!("Expected an array"),
        };

        let expected_dates = vec![
            Some(NaiveDate::from_ymd_opt(2024, 3, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 6, 1)),
            Some(NaiveDate::from_ymd_opt(2024, 2, 1)),
        ];
        let expected_days: Vec<Option<i32>> = expected_dates
            .into_iter()
            .map(|arg0: Option<Option<NaiveDate>>| date_to_days(arg0.flatten()))
            .collect();

        let result_as_date32 =
            result_array.as_any().downcast_ref::<Date32Array>().unwrap();

        let expected_array = Date32Array::from(expected_days);

        assert_eq!(result_as_date32, &expected_array);
    }
}
