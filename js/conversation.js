const {ipcRenderer} = require('electron')
window.$ = window.jQuery = require('jquery')

let logout = document.getElementById('btn-logout')
let textWindow = document.getElementById('translation')
let canvas = document.getElementById('video-stream-canvas')

let ctx = canvas.getContext('2d');
let mediaStream = null;
let video = document.getElementById('camera')
let cameraShots = new Array()

let canvasMasked = document.getElementById('masked-window-canvas')
let canvasMaskedWindow = document.getElementById('masked-window')
let ctxMasked = canvasMasked.getContext('2d');


let canvasCharPrediction = document.getElementById('char-prediction-window-canvas')
let canvasCharPredictionWindow = document.getElementById('char-prediction-window')
let ctxCharPrediction = canvasCharPrediction.getContext('2d');


let canvasGestureClassification = document.getElementById('gesture-classification-window-canvas')
let canvasGestureClassificationWindow = document.getElementById('gesture-classification-window')
let ctxGestureClassification = canvasGestureClassification.getContext('2d');


let canvasFinalPrediction = document.getElementById('final-prediction-window-canvas')
let canvasFinalPredictionWindow = document.getElementById('final-prediction-window')
let ctxFinalPrediction = canvasFinalPrediction.getContext('2d');


let maskHeight = canvasMaskedWindow.offsetHeight
let maskWidth = canvasMaskedWindow.offsetWidth

const DEBUG = true

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
  $.ajax({
    type: "POST",
    url: "http://localhost:5000/",
    data: {'img_array': JSON.stringify(cameraShots), 'debug': DEBUG},
    success: function(response) {
      processResponse(response)
    },
    error: function(msg) {
      console.log(msg)
    }
  })
}

function processResponse(response) {
  textWindow.innerHTML = textWindow.innerHTML + response.result;
  if (DEBUG) {
    assignImgToContext(ctxMasked, response.resized)
    assignImgToContext(ctxCharPrediction, response.predictedChars)
    assignImgToContext(ctxGestureClassification, response.classifiedGestures)
    assignImgToContext(ctxFinalPrediction, response.finallyPredicted)
  }
  scrollToBottom(1000, textWindow);
}

function assignImgToContext(context, responsePart) {
  var img = new Image
  img.onload = function() {
    context.drawImage(img, 0, 0, maskWidth, maskHeight)
  }
  img.src = responsePart
}

function snapshot() {
  if (video.videoWidth == 0) {
    return;
  }
  if (canvas.width != video.videoWidth) {
    canvas.width = maskWidth
    canvas.height = maskHeight
    if (DEBUG) {
      insertBorderToWindow(canvasMaskedWindow)
      insertBorderToWindow(canvasGestureClassificationWindow)
      insertBorderToWindow(canvasCharPredictionWindow)
      insertBorderToWindow(canvasFinalPredictionWindow)
    }
  }
  if (mediaStream) {
    if (cameraShots.length >= 5) {
      classifyLetters()
      cameraShots = new Array()
    }
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight,
                         0, 0, canvas.width, canvas.height);
    cameraShots.push(canvas.toDataURL('image/png'));
  }
}

function insertBorderToWindow(canvasWindow) {
  canvasWindow.style.border = "1px solid gray"
}

function init () {
  resizeCanvas(canvasMasked, maskWidth, maskHeight)
  resizeCanvas(canvas, maskWidth, maskHeight)
  resizeCanvas(canvasCharPrediction, maskWidth, maskHeight)
  resizeCanvas(canvasGestureClassification, maskWidth, maskHeight)
  resizeCanvas(canvasFinalPrediction, maskWidth, maskHeight)
}

function resizeCanvas(canvas, width, height) {
  canvas.width = width
  canvas.height = height
}

init()
setInterval(snapshot, 200) // 5 frames per second
