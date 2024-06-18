let recognizer;
// import {loadGraphModel} from '@tensorflow/tfjs-converter';
// import * as tf from '@tensorflow/tfjs';
// const tf = require('@tensorflow/tfjs');

// запускаем модель
async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    // Add this line.
    buildModel();
   }
// One frame is ~23ms of audio.
const NUM_FRAMES = 100;
const frameSize = 232; 
let examples = [];
let examples1 = [];


//собирает информацию с микрофона, работает
function collect_left(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   console.log(frameSize);
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label}); 
   document.querySelector('#example-counter').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
 console.log(examples);
}

function collect_right(label) {
    if (recognizer.isListening()) {
      return recognizer.stopListening();
    }
    if (label == null) {
      return;
    }
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
      let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      examples.push({vals, label}); 
      document.querySelector('#example-counter').textContent =
          `${examples.length} examples collected`;
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
   }

function collect_noise(label) {
    if (recognizer.isListening()) {
      return recognizer.stopListening();
    }
    if (label == null) {
      return;
    }
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
      let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      examples.push({vals, label}); 
      document.querySelector('#example-counter').textContent =
          `${examples.length} examples collected`;
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
   }



//конец микрофона

//



//1st try
//try collect data from files

//изменил const на let 
// const leftExamples = [];
// const rightExamples = [];
// const noiseExamples = [];

// async function submit_data() {
// document.getElementById('upload-left').addEventListener('change', (event) => {
//   const files = event.target.files;
//   for (let i = 0; i < files.length; i++) {
//     const file = files[i];
//     const reader = new FileReader();
//     reader.onload = async (event) => {
//       const audioBuffer = await decodeAudioData(event.target.result);
//       leftExamples.push(audioBuffer);
//       updateExampleCounter();
//     };
//     reader.readAsArrayBuffer(file);
//   }
// });

// document.getElementById('upload-right').addEventListener('change', (event) => {
//   const files = event.target.files;
//   for (let i = 0; i < files.length; i++) {
//     const file = files[i];
//     const reader = new FileReader();
//     reader.onload = async (event) => {
//       const audioBuffer = await decodeAudioData(event.target.result);
//       rightExamples.push(audioBuffer);
//       updateExampleCounter();
//     };
//     reader.readAsArrayBuffer(file);
//   }
// });


// document.getElementById('upload-noise').addEventListener('change', (event) => {
//   const files = event.target.files;
//   for (let i = 0; i < files.length; i++) {
//     const file = files[i];
//     const reader = new FileReader();
//     reader.onload = async (event) => {
//       const audioBuffer = await decodeAudioData(event.target.result);
//       noiseExamples.push(audioBuffer);
//       updateExampleCounter();
//     };
//     reader.readAsArrayBuffer(file);
//   }
// });
// }
// async function decodeAudioData(arrayBuffer) {
//   const audioContext = new AudioContext();
//   const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
//   return audioBuffer;
// }

// function updateExampleCounter() {
//   const exampleCounter = document.getElementById('example-counter');
//   exampleCounter.innerText = `${leftExamples.length + rightExamples.length + noiseExamples.length} examples collected`;
//   console.log(leftExamples.length, rightExamples.length, noiseExamples.length);
//   console.log(leftExamples);
// }

// end
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;
// // трениноровка модели с датасетом
// // загружаем данные
// async function train(leftExamples, rightExamples, noiseExamples) {
//   toggleButtons(false);
  
//   // Combine all examples into a single array
//   console.log(leftExamples);
//   console.log(rightExamples);
//   console.log(noiseExamples);
//   console.log(leftExamples.length.cocncat(rightExamples.length));
//   const allExamples = leftExamples.concat(rightExamples).concat(noiseExamples);

//   // Convert the examples into tensors
//   const ys = tf.oneHot(allExamples.map(example => example.label), 3);
//   const xsShape = [allExamples.length, ...INPUT_SHAPE];
//   const xs = tf.tensor(allExamples.map(example => example.vals).flat(), xsShape);

//   // Train the model
//   await model.fit(xs, ys, {
//     batchSize: 16,
//     epochs: 10,
//     callbacks: {
//       onEpochEnd: (epoch, logs) => {
//         document.querySelector('#accuracy_weight').textContent =
//           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
//       }
//     }
//   });

//   toggleButtons(true);
// }

//датасет через файлы
document.getElementById('upload-left').addEventListener('change', handleFileUpload);

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    return;
  }

  const arrayBuffer = await file.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  console.log(arrayBuffer)
  console.log(audioContext)
  console.log(audioBuffer)

  // Пример: получение данных спектрограммы
  const channelData = audioBuffer.getChannelData(0);
  let data = []; // Здесь будет храниться спектрограмма
  
  // Получаем данные спектрограммы
  for (let i = 0; i < channelData.length; i += frameSize) {
    const frame = channelData.slice(i, i + frameSize);
    data.push(...frame);
  }

  let vals = normalize1(data.slice(-frameSize * NUM_FRAMES));
  console.log(vals);
  let counter = 0;
  examples1.push({vals, label: 0}); 
  document.querySelector('#example-counter').textContent =
      `${examples1.length} examples collected`;
      examples1.forEach(element => {
        element.vals = new Float32Array(element.vals);
        element.label = counter;
        counter++;
      });
      console.log(examples1);
  console.log(examples1);
}


function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}
function normalize1(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
 }


// leftExamples

// тренировка модели с микрофоном
async function train() {
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);
  // const ys = tf.oneHot(leftExamples.map(e => e.label), 3);
  // const xsShape = [leftExamples.length, ...INPUT_SHAPE];
  // const xs = tf.tensor(flatten(leftExamples.map(e => e.vals)), xsShape);
  console.log(examples);

 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#accuracy_weight').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

//не помню что за функция тренировки
// async function train() {
//   toggleButtons(false);

//   // Подготовка данных
//   const ys = tf.oneHot(tf.tensor(examples.map(e => e.label), 'int32'), 3);
//   const xsShape = [examples.length, ...INPUT_SHAPE];
//   const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

//   // Обучение модели
//   await model.fit(xs, ys, {
//     batchSize: 16,
//     epochs: 10,
//     callbacks: {
//       onEpochEnd: (epoch, logs) => {
//         document.querySelector('#accuracy_weight').textContent =
//           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
//       }
//     }
//   });

//   // Очистка ресурсов
//   tf.dispose([xs, ys]);
//   toggleButtons(true);
// }

// смешная тренировка, что вечно выдает ошибки

async function train1() {
  toggleButtons(false);

  console.log(examples1)
  true_examples1 = examples1;
  console.log(true_examples1)
  // Подготовка данных
  const ys = tf.oneHot(true_examples1.map(e => e.label), 3);
  const xsShape = [true_examples1.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(true_examples1.map(e => e.vals)), xsShape);
  console.log(true_examples1);

  // Обучение модели
  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector('#accuracy_weight').textContent =
          `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
      }
    }
  });
  toggleButtons(true);
}

//построение модели
function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

// работает - не трогай!!!
function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

function flatten2(arr) {
  return arr.reduce((flat, toFlatten) => {
    return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
  }, []);
}

// работает - не трогай!!!
async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    document.getElementById('prediction').textContent = label;
    if (label == 2) {
      return;
    }
    let delta = 0.1;
    const prevValue = +document.getElementById('output').value;
    document.getElementById('output').value =
        prevValue + (label === 0 ? -delta : delta);
   }
   
   // работает - не трогай!!!
   function listen() {
    if (recognizer.isListening()) {
      recognizer.stopListening();
      toggleButtons(true);
      document.getElementById('listen').textContent = 'Listen';
      return;
    }
    toggleButtons(false);
    document.getElementById('listen').textContent = 'Stop';
    document.getElementById('listen').disabled = false;
   
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
      const probs = model.predict(input);
      const predLabel = probs.argMax(1);
      await moveSlider(predLabel);
      tf.dispose([input, probs, predLabel]);
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
    console.log(model);
   }
   
// работает - не трогай!!!
async function save_weights() {
  model.save('downloads://');
  console.log('Model saved as files in downloads, model name \"alpha\"');
}

// //сохранение через браузер
// async function save_weights_browser() {
  // model.save('localstorage://alpha');
//   const saveResults = await model.save('localstorage://alpha');
//   console.log('Model saved as files in downloads, model name \"alpha\"');
//   console.log(saveResults)
// }

// const model = tf.sequential(
//   {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
// console.log('Prediction from original model:');
// model.predict(tf.ones([1, 3])).print();

// const saveResults = await model.save('localstorage://my-model-1');

// const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
// console.log('Prediction from loaded model:');
// loadedModel.predict(tf.ones([1, 3])).print();


// async function load_weights() {
//   const model = await tf.loadLayersModel('localstorage://alpha');
//   console.log('Model kys"');
// }
// document.getElementById('model-file').addEventListener('change', async (event) => {
//   const files = event.target.files;
//   const jsonFile = Array.from(files).find(file => file.name.endsWith('.json'));
//   const weightsFiles = Array.from(files).filter(file => file.name.endsWith('.bin'));

//   const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...weightsFiles]));
//   console.log('Model loaded from files');
//   model.summary();
// });

// document.getElementById('model-file').addEventListener('change', async (event) => {
//   const files = event.target.files;
//   const jsonFile = Array.from(files).find(file => file.name.endsWith('.json'));
//   const weightsFiles = Array.from(files).filter(file => file.name.endsWith('.bin'));

//   const uploadJSONInput = document.getElementById('upload-json');
//   const uploadWeightsInput = document.getElementById('upload-weights');
//   const model = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
//   console.log('Model loaded from files');
//   model.summary();
//   console.log(model);
// });

// async function load_weights_browser() {
  // const model = await tf.loadLayersModel('localstorage://alpha');
  // const loadedModel = await tf.loadLayersModel('localstorage://alpha');
  // const model = await loadGraphModel('localstorage://alpha');
  // console.log(loadedModel)
  // model.summary();
// }

// работает - не трогай!!!
async function submit_files(){
  const uploadJSONInput = document.getElementById('upload-json');
  const uploadWeightsInput = document.getElementById('upload-weights');
  model = await tf.loadLayersModel(tf.io.browserFiles(
     [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
  console.log('Model loaded from files');
  console.log(model);
  console.log(uploadJSONInput);
  console.log(uploadWeightsInput);

  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
}
  
app();