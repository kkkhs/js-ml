const IMG_URL_PREFIX = "http://127.0.0.1:8888/brand/train/";

const loadImg = (src) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = src;
    img.width = 224;
    img.height = 224;
    img.onload = () => {
      resolve(img);
    };
  });
};

export const getInputs = async () => {
  const loadImgs = [];
  const labels = [];
  ["android", "apple", "windows"].forEach(async (type) => {
    for (let i = 0; i < 30; i++) {
      const src = `${IMG_URL_PREFIX}${type}-${i}.jpg`;
      const imgP = loadImg(src);
      loadImgs.push(imgP);
      labels.push([
        type === "android" ? 1 : 0,
        type === "apple" ? 1 : 0,
        type === "windows" ? 1 : 0,
      ]);
    }
  });

  const inputs = await Promise.all(loadImgs);

  return { inputs, labels };
};
