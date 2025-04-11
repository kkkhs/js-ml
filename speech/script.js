// import * as tf from "@tensorflow/tfjs";
// import * as tfvis from "@tensorflow/tfjs-vis";
// import * as speechCommands from "@tensorflow-models/speech-commands";

const MODEL_PATH = "http://127.0.0.1:8888/speech";

window.onload = async () => {
  const recognizer = speechCommands.create(
    "BROWSER_FFT",
    null,
    MODEL_PATH + "/model.json", // 模型信息
    MODEL_PATH + "/metadata.json" // 源信息
  );

  await recognizer.ensureModelLoaded();

  const labels = recognizer.wordLabels(); // 可识别的标签
  const resultEl = document.querySelector("#result");
  resultEl.innerHTML = labels.map((label) => `<div>${label}</div>`).join("");

  recognizer.listen(
    (result) => {
      const { scores } = result;
      const macValue = Math.max(...scores);
      const index = scores.indexOf(macValue);
      const resultEl = document.querySelector("#result");
      resultEl.innerHTML = labels
        .map((label, i) => {
          if (i === index) {
            return `<div style="background-color: aqua;">${label}</div>`;
          }
          return `<div>${label}</div>`;
        })
        .join("");
    },
    {
      overlapFactor: 0.3, // 识别的间隔度
      probabilityThreshold: 0.75, // 识别的置信度
    }
  );
};
