import * as tf from "@tensorflow/tfjs";

// 传统 for 循环
const input = [1, 2, 3, 4];
const w = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
  [4, 5, 6, 7],
];
const output = [0, 0, 0, 0];

for (let i = 0; i < w.length; i++) {
  for (let j = 0; j < input.length; j++) {
    output[i] += input[j] * w[i][j];
  }
}

console.log(output);

// 使用 tensor 计算神经网络
tf.tensor(w).dot(tf.tensor(input)).print();
