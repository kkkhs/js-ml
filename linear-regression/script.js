// import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

const xs = [1, 2, 3, 4];
const ys = [1, 3, 5, 7];

window.onload = async () => {
  tfvis.render.scatterplot(
    {
      name: "线性回归训练集",
    },
    {
      values: xs.map((x, i) => ({ x, y: ys[i] })),
    },
    {
      xAxisDomain: [0, 5],
      yAxisDomain: [0, 10],
    }
  );

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tf.losses.meanSquaredError, // 设置损失函数为：均方误差
    optimizer: tf.train.sgd(0.1), // 设置优化器为：随机梯度下降（学习率0.1）
  });

  const inputs = tf.tensor(xs);
  const labels = tf.tensor(ys);
  await model.fit(inputs, labels, {
    batchSize: 1, // 小批量
    epochs: 100, //训练迭代次数
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]),
  });
};
