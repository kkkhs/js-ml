import { getInputs } from "./data";
import { img2x, file2img } from "./utils";
import * as tf from "@tensorflow/tfjs";
// import * as tfvis from "@tensorflow/tfjs-vis";

const MOBILENET_MODEL_PATH =
  "http://127.0.0.1:8888/mobilenet/web_model/model.json";

window.onload = async () => {
  const { inputs, labels } = await getInputs();
  const surface = tfvis
    .visor()
    .surface({ name: "输入示例", styles: { height: 250 } });
  inputs.forEach((imgEl, index) => {
    surface.drawArea.appendChild(imgEl);
  });

  const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  //   mobilenet.summary();

  const layer = mobilenet.getLayer("conv_pw_13_relu");
  // 截断模型：截断到 conv_pw_13_relu 层
  const truncatedMobilenet = tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output,
  });

  const model = tf.sequential();
  // reshape 层：将截断后的模型输出的 7 7 1024 特征图转换为一维的 7 * 7 * 1024
  model.add(
    tf.layers.flatten({
      inputShape: layer.outputShape.slice(1), // 第一位为 null
    })
  );
  // 全连接进行特征降维（7 7 1024 → 10）
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "relu",
    })
  );
  // 多分类层
  model.add(
    tf.layers.dense({
      units: 3,
      activation: "softmax",
    })
  );

  model.compile({
    loss: "categoricalCrossentropy", // 多分类交叉熵
    optimizer: tf.train.adam(),
  });

  const { xs, ys } = tf.tidy(() => {
    const xs = tf.concat(
      inputs.map((img) => truncatedMobilenet.predict(img2x(img)))
    );
    const ys = tf.tensor(labels);
    console.log(xs, ys);
    return { xs, ys };
  });

  await model.fit(xs, ys, {
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks({ name: "训练效果" }, ["loss"], {
      callbacks: ["onEpochEnd"],
    }),
  });

  window.predict = async (file) => {
    const img = await file2img(file);
    document.body.appendChild(img);
    const pred = tf.tidy(() => {
      const input = img2x(img);
      return model.predict(truncatedMobilenet.predict(input));
    });

    const index = pred.argMax(1).dataSync()[0];
    setTimeout(() => {
      alert(`预测结果: ${["android", "apple", "windows"][index]}`);
    }, 0);
  };
};
