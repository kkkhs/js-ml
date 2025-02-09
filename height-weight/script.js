// import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

window.onload = async () => {
  const heights = [150, 160, 170];
  const weights = [40, 50, 60];

  tfvis.render.scatterplot(
    {
      name: "身高体重训练数据",
    },
    { values: heights.map((x, i) => ({ x, y: weights[i] })) },
    {
      xAxisDomain: [140, 180],
      yAxisDomain: [30, 70],
    }
  );

  // 归一化: 利用 tensor 数学运算
  const inputs = tf.tensor(heights).sub(150).div(20);
  const labels = tf.tensor(weights).sub(40).div(20);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1),
  });

  await model.fit(inputs, labels, {
    batchSize: 2,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]),
  });

  // 预测时也需要进行归一化
  const output = model.predict(tf.tensor([180]).sub(150).div(20));
  // 预测结果反归一化
  alert(
    `如果身高为 180 cm，那么预测体重 为 ${
      output.mul(20).add(40).dataSync()[0]
    } kg`
  );
};
