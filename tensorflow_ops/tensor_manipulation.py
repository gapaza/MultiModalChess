import tensorflow as tf
"""
  _______                               __  __                _                _         _    _               
 |__   __|                             |  \/  |              (_)              | |       | |  (_)              
    | |  ___  _ __   ___   ___   _ __  | \  / |  __ _  _ __   _  _ __   _   _ | |  __ _ | |_  _   ___   _ __  
    | | / _ \| '_ \ / __| / _ \ | '__| | |\/| | / _` || '_ \ | || '_ \ | | | || | / _` || __|| | / _ \ | '_ \ 
    | ||  __/| | | |\__ \| (_) || |    | |  | || (_| || | | || || |_) || |_| || || (_| || |_ | || (_) || | | |
    |_| \___||_| |_||___/ \___/ |_|    |_|  |_| \__,_||_| |_||_|| .__/  \__,_||_| \__,_| \__||_| \___/ |_| |_|
                                                                | |                                           
                                                                |_|                                           
    - tf.reshape:     Reshapes a tensor.
    - tf.transpose:   Transposes a tensor.
    - tf.squeeze:     Removes dimensions of size 1 from the tensor.
    - tf.expand_dims: Adds a new axis to the tensor.
    - tf.concat:      Concatenates tensors along a specified axis.
    - tf.slice:       Extracts a slice from a tensor.
    - tf.split:       Splits a tensor into multiple sub-tensors.
"""

manip_tensor = [
    [1, 1, 1],
    [3, 4, 5],
]
reshape_tensor = [
    [1, 1],
    [1, 3],
    [4, 5],
]
transpose_tensor = [
    [1, 3],
    [1, 4],
    [1, 5],
]

def reshape():
    reshaped_tensor = tf.reshape(manip_tensor, [3, 2])
    print('Reshaped:', reshaped_tensor)

def transpose():
    transposed_tensor = tf.transpose(manip_tensor)
    print('Transposed:', transposed_tensor)


squeeze_tensor = [
    [1],
    [2],
]
squeeze_tensor_2 = [
    [1, 2],
]

def squeeze():
    squeezed_tensor = tf.squeeze(squeeze_tensor_2, axis=0)
    print('Squeezed:', squeezed_tensor)



expand_dims_tensor = [1, 2]

def expand_dims():
    expanded_tensor = tf.expand_dims(expand_dims_tensor, axis=0)
    print('Expanded:', expanded_tensor)


concat_tensor_1 = [
    [1, 2],
    [3, 4],
]

concat_tensor_2 = [
    [5, 6],
    [7, 8],
]

def concat():
    concat_tensor = tf.concat([concat_tensor_1, concat_tensor_2], axis=1)
    print('Concatenated:', concat_tensor)


slice_tensor = [
    [1, 2, 3],
    [4, 5, 6],
]

def slice():
    s_tensor = tf.constant(slice_tensor, dtype=tf.int32)

    begin_slice = [1, 0]  # slice offset in each dimension
    size_slice = [1, 3]   # slice size in each dimension
    sliced_tensor_1 = tf.slice(s_tensor, begin_slice, size_slice)
    print('Sliced:', sliced_tensor_1)

    # Can also slice with tf.Tensor.getitem aka pythonic slicing
    sliced_tensor_2 = s_tensor[1:, :]
    print('Sliced:', sliced_tensor_2)







if __name__ == "__main__":
    # reshape()
    # transpose()
    # squeeze()
    # expand_dims()
    # concat()
    slice()

