import tensorflow as tf
"""
  _                    _              _    ____                            _    _                    
 | |                  (_)            | |  / __ \                          | |  (_)                   
 | |      ___    __ _  _   ___  __ _ | | | |  | | _ __    ___  _ __  __ _ | |_  _   ___   _ __   ___ 
 | |     / _ \  / _` || | / __|/ _` || | | |  | || '_ \  / _ \| '__|/ _` || __|| | / _ \ | '_ \ / __|
 | |____| (_) || (_| || || (__| (_| || | | |__| || |_) ||  __/| |  | (_| || |_ | || (_) || | | |\__ \
 |______|\___/  \__, ||_| \___|\__,_||_|  \____/ | .__/  \___||_|   \__,_| \__||_| \___/ |_| |_||___/
                 __/ |                           | |                                                 
                |___/                            |_|                                                 
    
    Element-wise comparison operations
    - tf.equal
    - tf.not_equal
    - tf.less
    - tf.less_equal
    - tf.greater
    - tf.greater_equal
    
    Element-wise logical operations
    tf.logical_and
    tf.logical_or
    tf.logical_not

    Returns elements from two tensors depending on a given condition
    tf.where: returns the indices of non-zero elements, or multiplexes x and y.
    
"""


def equal():
    tensor_1 = [2, 2, 2]
    tensor_2 = [4, 2, 4]
    eq_op = tf.equal(tensor_1, tensor_2)
    print('Equal:', eq_op)



def logical_and():
    tensor_1 = [True, False, True]
    tensor_2 = [False, False, True]
    land_op = tf.logical_and(tensor_1, tensor_2)
    print('Logical AND:', land_op)


def where():
    """
    This operation has two modes:

        Return the indices of non-zero elements - When only condition is provided the result is an int64 tensor where
        each row is the index of a non-zero element of condition.
        The result's shape is [tf.math.count_nonzero(condition), tf.rank(condition)].

        Multiplex x and y - When both x and y are provided the result has the shape of x, y, and condition broadcast
        together. The result is taken from x where condition is non-zero or y where condition is zero.

    """

    # Mode 1
    tensor_1 = [False, False, True, True]
    positions = tf.where(tensor_1)
    print('Positions:', positions)

    tensor_2 = [[1, 0, 0], [1, 0, 1]]
    response = tf.where(tensor_2)
    print('Response:', response)

    # Mode 2: condition must have dtype bool
    # condition tensor acts as a mask, telling the func which values to pick from x (if True) or y (if False)
    response = tf.where(
        [True, False, False, True],
        [1, 2, 3, 4],
        [100, 200, 300, 400]
    )
    print('Response:', response)










if __name__ == "__main__":
    # equal()
    # logical_and()
    where()

