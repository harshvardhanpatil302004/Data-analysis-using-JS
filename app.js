document.getElementById("csvUpload").addEventListener("change", handleFile);

async function handleFile(event) {
  const file = event.target.files[0];
  const text = await file.text();
  
  console.log("Raw CSV Data:", text); // Debugging CSV content
  
  const data = Papa.parse(text, { header: true, dynamicTyping: true }).data;

  console.log("Parsed Data:", data); // Debugging parsed data
  
  const filtered = data.filter(
    row =>
      row.SalePrice &&
      row.LotArea &&
      row.OverallQual &&
      row.GrLivArea &&
      row.GarageCars &&
      row.TotalBsmtSF
  );

  console.log("Filtered Data Length:", filtered.length); // Debugging filter success

  if (filtered.length === 0) {
    console.error("No valid rows found after filtering. Check CSV column names.");
    return;
  }

  const features = filtered.map(row => [
    row.LotArea,
    row.OverallQual,
    row.GrLivArea,
    row.GarageCars,
    row.TotalBsmtSF
  ]);

  const labels = filtered.map(row => row.SalePrice);

  const featureTensor = tf.tensor2d(features);
  const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

  console.log("Feature Tensor:", featureTensor.arraySync()); // Debugging tensor conversion
  console.log("Label Tensor:", labelTensor.arraySync());

  const [trainX, testX, trainY, testY] = tf.tidy(() => {
    const splitIndex = Math.floor(0.7 * featureTensor.shape[0]);
    return [
      featureTensor.slice(0, splitIndex),
      featureTensor.slice(splitIndex),
      labelTensor.slice(0, splitIndex),
      labelTensor.slice(splitIndex)
    ];
  });

  console.log("Train/Test Shapes:", trainX.shape, testX.shape, trainY.shape, testY.shape); // Debugging dataset split

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [5], units: 1 }));
  model.compile({ optimizer: "adam", loss: "meanSquaredError" });

  await model.fit(trainX, trainY, {
    epochs: 100,
    batchSize: 32,
    shuffle: true
  });

  // Generate predictions for both training and testing sets
  const trainPreds = model.predict(trainX).flatten();
  const testPreds = model.predict(testX).flatten();

  const trainPredArr = await trainPreds.array();
  const testPredArr = await testPreds.array();

  const trainActualArr = await trainY.array();
  const testActualArr = await testY.array();

  console.log("Train Predictions:", trainPredArr);
  console.log("Train Actual:", trainActualArr);
  console.log("Test Predictions:", testPredArr);
  console.log("Test Actual:", testActualArr);

  // Compute metrics for training and testing datasets
  const trainMAE = trainPredArr.reduce((acc, pred, i) => acc + Math.abs(pred - trainActualArr[i]), 0) / trainPredArr.length;
  const testMAE = testPredArr.reduce((acc, pred, i) => acc + Math.abs(pred - testActualArr[i]), 0) / testPredArr.length;

  // Corrected HTML injection with template literals
  document.getElementById("metrics").innerHTML = `
    <p><strong>Training MAE:</strong> ${trainMAE.toFixed(2)}</p>
    <p><strong>Testing MAE:</strong> ${testMAE.toFixed(2)}</p>
  `;

  new Chart(document.getElementById("predictionChart"), {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Training Data",
          data: trainPredArr.map((p, i) => ({ x: trainActualArr[i], y: p })),
          backgroundColor: "rgba(255, 99, 132, 0.6)"
        },
        {
          label: "Testing Data",
          data: testPredArr.map((p, i) => ({ x: testActualArr[i], y: p })),
          backgroundColor: "rgba(75, 192, 192, 0.6)"
        }
      ]
    },
    options: {
      scales: {
        x: { title: { display: true, text: "Actual Price" } },
        y: { title: { display: true, text: "Predicted Price" } }
      }
    }
  });

  // Dispose of tensors to free memory
  featureTensor.dispose();
  labelTensor.dispose();
  trainX.dispose();
  testX.dispose();
  trainY.dispose();
  testY.dispose();
}
