// import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

import { getData } from "./data";

window.onload = async () => {
  const data = getData(400);
  console.log(data);

  tfvis.render.scatterplot(
    { name: "逻辑回归训练数据" },
    {
      values: [
        data.filter((v) => v.label === 1),
        data.filter((v) => v.label === 0),
      ],
    },
    {
      xAxisDomain: [-6, 6],
      yAxisDomain: [-6, 6],
    }
  );

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [2], // 点
      activation: "sigmoid", // 激活函数（限制output 在 [0, 1])
    })
  );

  model.compile({
    loss: tf.losses.logLoss, // 对数损失
    optimizer: tf.train.adam(0.1),
  });

  const inputs = tf.tensor(data.map((i) => [i.x, i.y]));
  const labels = tf.tensor(data.map((i) => i.label));

  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: "训练过程",
      },
      ["loss"]
    ),
  });

  window.predict = (form) => {
    const targetV = tf.tensor([[form.x.value * 1, form.y.value * 1]]);
    const pred = model.predict(targetV);
    alert(`预测结果：${pred.dataSync()[0]}`);
  };
};
