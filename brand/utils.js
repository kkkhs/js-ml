export function img2x(imgEl) {
  return tf.tidy(() => {
    return (
      tf.browser
        .fromPixels(imgEl)
        // 归一化：
        .toFloat()
        .div(255 / 2)
        .sub(1)
        // 调整形状：
        .reshape([1, 224, 224, 3])
    );
  });
}

export function file2img(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file); // readAsDataURL
    reader.onload = (e) => {
      // 文件转图片
      const img = document.createElement("img");
      img.src = e.target.result;
      img.width = 224;
      img.height = 224;
      img.onload = () => {
        resolve(img);
      };
    };
  });
}
