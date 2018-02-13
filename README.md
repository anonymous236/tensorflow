# tensorflow
## 基本使用
使用 TensorFlow, 你必须明白 TensorFlow:
* 使用图 (graph) 来表示计算任务.
* 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
* 使用 tensor 表示数据.
* 通过 变量 (Variable) 维护状态.
* 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

## 综述
TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 <font color=#00ffff size=72>op</font> (operation 的缩写). 一个 op 获得 0 个或多个 Tensor, 执行计算, 产生 0 个或多个 Tensor. 每个 Tensor 是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 [batch, height, width, channels].

一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在 会话 里被启动. 会话 将图的 op 分发到诸如 CPU 或 GPU 之类的 设备 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. 在 Python 语言中, 返回的 tensor 是 numpy ndarray 对象; 在 C 和 C++ 语言中, 返回的 tensor 是 tensorflow::Tensor 实例.
