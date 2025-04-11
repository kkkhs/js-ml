import * as tf from "@tensorflow/tfjs";
// import * as tfvis from "@tensorflow/tfjs-vis";
import { getIrisData, IRIS_CLASSES } from "./data";

window.onload = async () => {
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 10,
      inputShape: [xTrain.shape[1]],
      activation: "sigmoid",
    })
  );

  model.add(
    tf.layers.dense({
      units: 3,
      activation: "softmax",
    })
  );

  model.compile({
    // 交叉熵损失函数
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.1),
    // 准确率
    metrics: ["accuracy"],
  });

  await model.fit(xTrain, yTrain, {
    epochs: 100,
    // 传入验证集
    validationData: [xTest, yTest],
    callbacks: tfvis.show.fitCallbacks(
      { name: "训练效果" },
      // 记录 损失、验证集损失、准确率、验证机准确率
      ["loss", "val_loss", "acc", "val_acc"],
      { callbacks: ["onEpochEnd"] } // 只在epoch结束记录
    ),
  });

  window.predict = (from) => {
    const input = tf.tensor([
      [from.a.value * 1, from.b.value * 1, from.c.value * 1, from.d.value * 1],
    ]);
    const pred = model.predict(input);

    // argMax(x) 输出x维中的最大值的坐标
    alert(`预测结果: ${IRIS_CLASSES[pred.argMax(1).dataSync()[0]]}`);
  };
};
