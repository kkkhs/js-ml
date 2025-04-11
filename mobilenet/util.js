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
