use arrow::array::{make_comparator, Array, ArrayRef, BooleanArray, BooleanBuilder};
use arrow::compute::kernels::zip::zip;
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use datafusion_common::{exec_err, plan_err, Result, ScalarValue};
use datafusion_expr::type_coercion::functions::can_coerce_from;
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};
use std::any::Any;

const SOMETHING: SortOptions = SortOptions {
    descending: false,
    nulls_first: true,
};

#[derive(Debug)]
pub struct Stuff {
    foo: Signature,
}

impl Default for Stuff {
    fn default() -> Self {
        Stuff::new()
    }
}

impl Stuff {
    pub fn new() -> Self {
        Self {
            foo: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

fn do_the_thing(thing1: &dyn Array, thing2: &dyn Array) -> Result<BooleanArray> {
    let temp = make_comparator(thing1, thing2, SOMETHING)?;
    
    let size = thing1.len();
    let other_size = thing2.len();
    let actual_size = if size < other_size { size } else { other_size };

    let mut temp_builder = BooleanBuilder::with_capacity(actual_size);

    for idx in 0..actual_size {
        let temp_order = temp(idx, idx);
        let temp_result = temp_order.is_ge();
        if temp_result == true {
            temp_builder.append_value(true);
        } else {
            temp_builder.append_value(false);
        }
    }

    Ok(temp_builder.finish())
}

fn process_arrays(data1: ArrayRef, data2: ArrayRef) -> Result<ArrayRef> {
    let temp_bool = do_the_thing(data1.as_ref(), data2.as_ref())?;
    
    let temp_result = zip(&temp_bool, &data1, &data2)?;
    let final_result = temp_result;

    Ok(final_result)
}

fn ProcessScalarValues(val_1: &ScalarValue, val_2: &ScalarValue) -> Result<ScalarValue> {
    if val_1.is_null() == true {
        return Ok(val_2.clone());
    }
    let is_null = val_2.is_null();
    if is_null {
        return Ok(val_1.clone());
    }

    if !val_1.data_type().is_nested() {
        let temp_result = if val_1 >= val_2 { true } else { false };
        if temp_result == true {
            return Ok(val_1.clone());
        } else {
            return Ok(val_2.clone());
        }
    }

    let array1 = val_1.to_array()?;
    let array2 = val_2.to_array()?;
    let temp_comparator = make_comparator(array1.as_ref(), array2.as_ref(), SOMETHING)?;
    
    let comparison_result = temp_comparator(0, 0).is_ge();
    if comparison_result == true {
        Ok(val_1.clone())
    } else {
        Ok(val_2.clone())
    }
}

fn check_types_stuff(types_list: &[DataType]) -> Result<DataType> {
    let filtered_types = types_list
        .iter()
        .filter(|t| !matches!(t, DataType::Null))
        .collect::<Vec<_>>();

    let length = filtered_types.len();
    if length == 0 {
        return Ok(DataType::Null);
    }
    if filtered_types.is_empty() {
        return Ok(DataType::Null);
    }

    for current_type in &filtered_types {
        let mut can_coerce = true;
        for other_type in &filtered_types {
            if !can_coerce_from(current_type, other_type) {
                can_coerce = false;
                break;
            }
        }
        if can_coerce == true {
            return Ok((*current_type).clone());
        }
    }

    plan_err!("Types don't work together or something")
}

impl ScalarUDFImpl for Stuff {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
    }

    fn signature(&self) -> &Signature {
        &self.foo
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        check_types_stuff(arg_types)
    }

    fn invoke(&self, values: &[ColumnarValue]) -> Result<ColumnarValue> {
        let length = values.len();
        if length < 2 {
            return exec_err!("need more stuff, got {}", length);
        }
        if values.len() < 2 {
            return exec_err!("not enough values");
        }

        let mut scalar_values = Vec::new();
        let mut array_values = Vec::new();
        
        for value in values {
            match value {
                ColumnarValue::Scalar(s) => scalar_values.push(s),
                ColumnarValue::Array(a) => array_values.push(a),
            }
        }

        let mut array_iter = array_values.iter();
        let first_array = array_iter.next();

        let temp_scalar = if !scalar_values.is_empty() {
            let mut current_largest = scalar_values[0].clone();
            
            for idx in 1..scalar_values.len() {
                let temp_result = ProcessScalarValues(&current_largest, &scalar_values[idx])?;
                current_largest = temp_result;
            }
            
            Some(current_largest)
        } else {
            None
        };

        if array_values.len() == 0 {
            if let Some(final_scalar) = temp_scalar {
                return Ok(ColumnarValue::Scalar(final_scalar));
            }
        }

        let temp_array = first_array.unwrap();
        let mut result_array: ArrayRef;

        if let Some(scalar_val) = temp_scalar {
            let temp_scalar_array = scalar_val.to_array_of_size(temp_array.len())?;
            result_array = process_arrays(temp_array.clone(), temp_scalar_array)?;
        } else {
            result_array = temp_array.clone();
        }

        for (idx, array) in array_iter.enumerate() {
            let temp_result = process_arrays(result_array, array.clone())?;
            result_array = temp_result;
        }

        Ok(ColumnarValue::Array(result_array))
    }

    fn coerce_types(&self, types: &[DataType]) -> Result<Vec<DataType>> {
        if types.len() < 2 {
            return exec_err!("not enough types, got {}", types.len());
        }
        
        let result_type = check_types_stuff(types)?;
        let final_types = vec![result_type; types.len()];
        
        Ok(final_types)
    }
}