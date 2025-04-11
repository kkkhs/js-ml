// import * as tf from "@tensorflow/tfjs";
import { file2img } from "./util";
import { IMAGENET_CLASSES } from "./imagenet_classes";

const MOBILENET_MODEL_PATH =
  "http://127.0.0.1:8888/mobilenet/web_model/model.json";

window.onload = async () => {
  const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

  window.predict = async (file) => {
    const img = await file2img(file);
    document.body.appendChild(img);
    const pred = tf.tidy(() => {
      const input = tf.browser
        .fromPixels(img)
        // 归一化：
        .toFloat()
        .div(255 / 2)
        .sub(1)
        // 调整形状：
        .reshape([1, 224, 224, 3]);

      return model.predict(input);
    });

    const index = pred.argMax(1).dataSync()[0];
    const className = IMAGENET_CLASSES[index];
    setTimeout(() => {
      alert(`预测结果: ${className}`);
    }, 0);
  };
};
