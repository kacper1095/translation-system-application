const {ipcRenderer} = require('electron');
window.$ = window.jQuery = require('jquery');

let logout = document.getElementById('btn-logout');
let textWindow = document.getElementById('translation');
let dictionaryPredictionWindow = document.getElementById('dictionary-prediction');
let topPredictionsWindow = document.getElementById('predicted');
let canvas = document.getElementById('video-stream-canvas');

let ctx = canvas.getContext('2d');
let mediaStream = null;
let video = document.getElementById('camera');
let cameraShots = new Array();

let canvasMasked = document.getElementById('masked-window-canvas');
let canvasMaskedWindow = document.getElementById('masked-window');
let ctxMasked = canvasMasked.getContext('2d');


let canvasCharPrediction = document.getElementById('char-prediction-window-canvas');
let canvasCharPredictionWindow = document.getElementById('char-prediction-window');
let ctxCharPrediction = canvasCharPrediction.getContext('2d');


let canvasGestureClassification = document.getElementById('gesture-classification-window-canvas');
let canvasGestureClassificationWindow = document.getElementById('gesture-classification-window');
let ctxGestureClassification = canvasGestureClassification.getContext('2d');


let canvasFinalPrediction = document.getElementById('final-prediction-window-canvas');
let canvasFinalPredictionWindow = document.getElementById('final-prediction-window');
let ctxFinalPrediction = canvasFinalPrediction.getContext('2d');


let maskHeight = canvasMaskedWindow.offsetHeight;
let maskWidth = canvasMaskedWindow.offsetWidth;

let predicted = true;
let initialPrediction = false;

const DEBUG = true;

let videoConstraints = {
  video: {
    optional: [
      {minWidth: 320},
      {minWidth: 640},
      {minWidth: 800},
      {minWidth: 900},
      {minWidth: 1024},
      {minWidth: 1280},
      {minWidth: 1920},
    ]
  }
}

logout.addEventListener('click', function(e) {
  ipcRenderer.send('ask-for-exit')
})

navigator.webkitGetUserMedia(videoConstraints,
  function(stream) {
    video.src = URL.createObjectURL(stream);
    mediaStream = stream;
  },
  function() {
    alert('Could not connect stream!');
  }
)

function isOverflowed(element) {
  return element.scrollHeight > element.clientHeight;
}

function scrollToBottom(scrollDuration, element) {
    $(element).stop().animate({
      scrollTop: element.scrollHeight
    }, scrollDuration, 'linear');
}

function classifyLetters(){
  console.log(cameraShots.length);
  $.ajax({
    type: "POST",
    url: "http://localhost:5000/",
    // async: false,
    data: {'img_array': JSON.stringify(cameraShots), 'debug': DEBUG},
    success: function(response) {
      if (!initialPrediction) {
        initialPrediction = true;
      } else {
        processResponse(response)
      }
      predicted = true;
    },
    error: function(msg) {
      console.log(msg);
      setTimeout(function() {
        classifyLetters();
      }, 50)
    }
  })
}

function processResponse(response) {
  if (response.result !== null) {
    textWindow.innerHTML = textWindow.innerHTML + response.result;
  }
  if (DEBUG) {
    assignImgIfNotNullToContext(ctxMasked, response.resized)
    assignImgIfNotNullToContext(ctxCharPrediction, response.predictedChars)
    assignImgIfNotNullToContext(ctxGestureClassification, response.classifiedGestures)
    assignImgIfNotNullToContext(ctxFinalPrediction, response.finallyPredicted)
    let insert = true
    let formattedResponse = []
    for (var index in response.nearestPredictions) {
      if (response.nearestPredictions[index] === "") {
        insert = false
        break
      } else {
        if (response.nearestPredictions[index] === " ") {
          formattedResponse[index] = "*space*"
        } else {
          formattedResponse[index] = response.nearestPredictions[index]
        }
      }
    }
    if (insert) {
      topPredictionsWindow.innerHTML = formattedResponse.join('<span style="display:inline-block; width: 1.7em;"></span>')
    }
    dictionaryPredictionWindow.innerHTML = dictionaryPredictionWindow.innerHTML + response.dictionaryPrediction
  }
  scrollToBottom(1000, textWindow);
}

function assignImgIfNotNullToContext(context, responsePart) {
  if (responsePart === null) return;
  var img = new Image;
  img.onload = function() {
    context.drawImage(img, 0, 0, maskWidth, maskHeight);
  }
  img.src = responsePart;
}

function snapshot() {
  if (video.videoWidth == 0) {
    return;
  }
  if (canvas.width != video.videoWidth) {
    canvas.width = maskWidth;
    canvas.height = maskHeight;
    if (DEBUG) {
      insertBorderToWindow(canvasMaskedWindow);
      insertBorderToWindow(canvasGestureClassificationWindow);
      insertBorderToWindow(canvasCharPredictionWindow);
      insertBorderToWindow(canvasFinalPredictionWindow);
    }
  }
  if (mediaStream) {
    if (predicted && cameraShots.length >= 1) {
      predicted = false;
      classifyLetters();
      cameraShots = Array();
    }
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight,
                         0, 0, canvas.width, canvas.height);
    if (predicted) {
      cameraShots.push(canvas.toDataURL('image/png'));
    }
  }
}

function insertBorderToWindow(canvasWindow) {
  canvasWindow.style.border = "1px solid gray";
}

function init () {
  resizeCanvas(canvasMasked, maskWidth, maskHeight);
  resizeCanvas(canvas, maskWidth, maskHeight);
  resizeCanvas(canvasCharPrediction, maskWidth, maskHeight);
  resizeCanvas(canvasGestureClassification, maskWidth, maskHeight);
  resizeCanvas(canvasFinalPrediction, maskWidth, maskHeight);
}

function resizeCanvas(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
}

init();
setInterval(snapshot, 1); // 5 frames per second
