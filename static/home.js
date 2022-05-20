const img = document.querySelector("#img");
const imgContainer = document.querySelector(".img");
const main = document.querySelector("main");
const button = document.querySelector("main .btn");
const prediction = document.querySelector(".prediction");
const loader = document.querySelector(".loader");
const imgNode = document.createElement("img");

let base64;

imgContainer.addEventListener("click", () => img.click());
imgNode.addEventListener("click", () => img.click());
img.onchange = async (e) => {
  try {
    base64 = await generateBase64FromImage(e.target.files[0]);

    typeBase64 = base64.split(";")[0].split(":")[1].split("/")[0];
    if (typeBase64 !== "image") return alert("File shoud be an image");

    imgNode.className = "img";
    imgNode.setAttribute("src", base64);
    if (main.hasChildNodes(imgContainer)) {
      imgContainer.remove();
      main.prepend(imgNode);
    }
    button.classList.remove("hidden");
    prediction.classList.remove("hidden");
  } catch (error) {
    console.log(error);
  }
};

button.addEventListener("click", async () => {
  if (!base64) return;
  button.classList.add("hidden");
  prediction.classList.add("hidden");
  loader.classList.remove("hidden");
  const res = await fetch("/identify", {
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      image: base64.split(",")[1],
    }),
    method: "POST",
  });
  const { preds } = await res.json();
  loader.classList.add("hidden");
  button.classList.remove("hidden");
  prediction.classList.remove("hidden");
  prediction.innerText = preds[0];
});

const generateBase64FromImage = (imageFile) => {
  const reader = new FileReader();
  const promise = new Promise((resolve, reject) => {
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (err) => reject(err);
  });

  reader.readAsDataURL(imageFile);
  return promise;
};
