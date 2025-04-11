import * as tf from "@tensorflow/tfjs";
// import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "../xor/data";

window.onload = async () => {
  const data = getData(200);
  tfvis.render.scatterplot(
    {
      name: "训练数据",
    },
    {
      values: [
        data.filter((p) => p.label === 1),
        data.filter((p) => p.label === 0),
      ],
    }
  );

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 10,
      inputShape: [2],
      activation: "tanh",
      // 1. 缓解过拟合：权重衰减法：l2正则化
      kernelRegularizer: tf.kernelRegularizer.l2({ l2: 1 }),
    })
  );
  //3. 缓解过拟合：提前停止训练法

  // 2. 缓解过拟合： 丢弃法
  model.add(tf.layers.dropout({ rate: 0.9 }));

  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1),
  });

  const inputs = tf.tensor(data.map((p) => [p.x, p.y]));
  const labels = tf.tensor(data.map((p) => p.label));

  await model.fit(inputs, labels, {
    validationData: 0.2,
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: "训练效果",
      },
      ["loss", "val_loss"],
      { callbacks: ["onEpochEnd"] }
    ),
  });

  window.predict = (from) => {};
};
