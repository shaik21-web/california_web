
let session;
async function loadModel() {
    session = await ort.InferenceSession.create("california_model.onnx");
    console.log("ONNX model loaded!");
}
function getInputVector() {
    return Float32Array.from([
        parseFloat(document.getElementById("MedInc").value),
        parseFloat(document.getElementById("HouseAge").value),
        parseFloat(document.getElementById("AveRooms").value),
        parseFloat(document.getElementById("AveBedrms").value),
        parseFloat(document.getElementById("Population").value),
        parseFloat(document.getElementById("AveOccup").value),
        parseFloat(document.getElementById("Latitude").value),
        parseFloat(document.getElementById("Longitude").value)
    ]);
}
async function predict() {
    const inputVector = getInputVector();
    const inputTensor = new ort.Tensor("float32", inputVector, [1, inputVector.length]);
    try {
        const results = await session.run({input: inputTensor});
        const output = results.output.data[0];
        document.getElementById("result").innerHTML = 
            `Predicted Median House Value: $${(output*100000).toFixed(2)}`;
    } catch(err) {
        console.error(err);
        document.getElementById("result").innerHTML = "Error running model.";
    }
}
document.getElementById("predictBtn").addEventListener("click", predict);
loadModel();
