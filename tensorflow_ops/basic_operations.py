import tensorflow as tf


"""
  ____               _          ____                            _    _
 |  _ \             (_)        / __ \                          | |  (_)
 | |_) |  __ _  ___  _   ___  | |  | | _ __    ___  _ __  __ _ | |_  _   ___   _ __   ___
 |  _ <  / _` |/ __|| | / __| | |  | || '_ \  / _ \| '__|/ _` || __|| | / _ \ | '_ \ / __|
 | |_) || (_| |\__ \| || (__  | |__| || |_) ||  __/| |  | (_| || |_ | || (_) || | | |\__ \
 |____/  \__,_||___/|_| \___|  \____/ | .__/  \___||_|   \__,_| \__||_| \___/ |_| |_||___/
                                      | |
                                      |_|
"""


#############################
### Constants / Variables ###
#############################

tensor_like_obj = [
    [1, 2, 3],
    [4, 5, 6]
]

def constants():
    constant = tf.constant(tensor_like_obj, dtype=tf.int64, name='constant')
    print(constant)  # shape: (2, 3)

def variables():
    variable = tf.Variable(tensor_like_obj, dtype=tf.int64, name='variable', trainable=True)
    print(variable)  # shape: (2, 3)




###############################
### Element-wise Operations ###
###############################

tensor_1 = [2, 2, 2]
tensor_2 = [4, 4, 4]

def add():
    add_op = tf.add(tensor_1, tensor_2)
    print('Addition:', add_op)


def subtract():
    sub_op = tf.subtract(tensor_1, tensor_2)
    print('Subtraction:', sub_op)

def multiply():
    mul_op = tf.multiply(tensor_1, tensor_2)
    print('Multiplication:', mul_op)

def divide():
    div_op = tf.divide(tensor_1, tensor_2)
    print('Division:', div_op)



############################
### Reduction Operations ###
############################
# - reduction operations along a specific axis
# tf.reduce_sum
# tf.reduce_mean
# tf.reduce_min
# tf.reduce_max
redux_tensor =  [
    [1, 1, 1],
    [3, 3, 3],
    [5, 5, 5]
]
# for axis=0
# reduce_sum: [9, 9, 9]
# reduce_mean: [3, 3, 3]
# reduce_min: [1, 1, 1]
# reduce_max: [5, 5, 5]
# for axis=1
# reduce_sum: [3, 9, 15]
# reduce_mean: [1, 3, 5]
# reduce_min: [1, 3, 5]
# reduce_max: [1, 3, 5]
redux_tensor_2 = [
    [[1, 1], [1, 1]],
    [[3, 3], [3, 3]],
    [[5, 5], [5, 9]]
]
# for axis=0
# reduce_sum: [[9, 9], [9, 13]]
# reduce_mean: [[3, 3], [3, 4]]
# reduce_min: [[1, 1], [1, 1]]
# reduce_max: [[5, 5], [5, 9]]
# for axis=1
# reduce_sum: [[2, 2], [6, 6], [10, 13]]
# reduce_mean: [[1, 1], [3, 3], [5, 6]]
# reduce_min: [[1, 1], [3, 3], [5, 5]]
# reduce_max: [[1, 1], [3, 3], [5, 9]]
# for axis=2
# reduce_sum: [[2, 2], [6, 6], [10, 14]]
# reduce_mean: [[1, 1], [3, 3], [5, 7]]
# reduce_min: [[1, 1], [3, 3], [5, 5]]
# reduce_max: [[1, 1], [3, 3], [5, 9]]


def reduce_sum():
    constant = tf.constant(redux_tensor_2, dtype=tf.int64, name='constant')
    sum_op = tf.reduce_sum(constant, axis=1, keepdims=False)
    print('Reduce Sum:', sum_op)

def reduce_mean():
    constant = tf.constant(redux_tensor_2, dtype=tf.int64, name='constant')
    mean_op = tf.reduce_mean(constant, axis=0, keepdims=False)
    print('Reduce Mean:', mean_op)

def reduce_min():
    constant = tf.constant(redux_tensor_2, dtype=tf.int64, name='constant')
    min_op = tf.reduce_min(constant, axis=0, keepdims=False)
    print('Reduce Min:', min_op)

def reduce_max():
    constant = tf.constant(redux_tensor_2, dtype=tf.int64, name='constant')
    max_op = tf.reduce_max(constant, axis=0, keepdims=False)
    print('Reduce Max:', max_op)







if __name__ == '__main__':
    reduce_sum()
    reduce_mean()
    reduce_min()
    reduce_max()
