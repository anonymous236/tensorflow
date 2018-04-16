## tf.transpose()
 * tf.transpose()作为数组的转置函数,原型如下:
 ```python
 def transpose(a, perm=None, name="transpose"):
  """Transposes `a`. Permutes the dimensions according to `perm`.
 ```
 * 令input_x 是一个 2x3x4的一个tensor, 假设perm = [1,0,2], 就是将最外2层转置,得到tensor应该是3x2x4的一个张量；<br>
 如果 perm=[0,2,1]说明要交换内层里面的两个维度,从原来的2x3x4变成2x4x3的张量
```python
input_x = [  
    [  
        [1, 2, 3, 4],  
        [5, 6, 7, 8],  
        [9, 10, 11, 12]  
    ],  
    [  
        [13, 14, 15, 16],  
        [17, 18, 19, 20],  
        [21, 22, 23, 24]  
    ]
]  
  
result0 = tf.transpose(input_x, perm=[1, 0, 2])
result1 = tf.transpose(input_x, perm=[0, 2, 1])
with tf.Session() as sess:
    print(sess.run(result0))
    print(sess.run(result1))
'''
>>> [[[ 1  2  3  4]
      [13 14 15 16]]
     [[ 5  6  7  8]
      [17 18 19 20]]
     [[ 9 10 11 12]
      [21 22 23 24]]]
      
    [[[ 1  5  9]
      [ 2  6 10]
      [ 3  7 11]
      [ 4  8 12]]
     [[13 17 21]
      [14 18 22]
      [15 19 23]
      [16 20 24]]]
'''
```
## tf.reshape()
 * tf.reshape(tensor, shape, name=None) 函数的作用是将tensor变换为参数shape的形式<br>
 shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算
 ```python
 # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                         [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

# -1 can also be used to infer the shape

# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                               [2, 2, 2],
                               [3, 3, 3]],
                              [[4, 4, 4],
                               [5, 5, 5],
                               [6, 6, 6]]]

# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
 ```

