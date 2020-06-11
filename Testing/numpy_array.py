# docs: An ndarray is a (usually fixed-size) multidimensional container 
# of items of the same type and size. The number of dimensions and items 
# in an array is defined by its shape, which is a tuple of N non-negative 
# integers that specify the sizes of each dimension. The type of items in 
# the array is specified by a separate data-type object (dtype), one of
# which is associated with each ndarray

import numpy as np

x = np.array([
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            ], np.int32)

#print(type(x))
#print(x.shape)  # gibt dimensionen des arrays als tupel inkl. enthaltene Itemanzahlen an, z.B (6, 13), also 2D Array
#print(x.dtype)  # gibt datentyp des arrays an


