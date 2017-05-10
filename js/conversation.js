const {ipcRenderer} = require('electron')
window.$ = window.jQuery = require('jquery')

let logout = document.getElementById('btn-logout')
let textWindow = document.getElementById('translation')
let canvas = document.getElementById('video-stream-canvas')

canvas.width = 320;
canvas.height = 240;

let ctx = canvas.getContext('2d');
let mediaStream = null;
let video = document.getElementById('camera')
let cameraShots = new Array()

let canvasMasked = document.getElementById('masked-window-canvas')
let canvasMaskedWindow = document.getElementById('masked-window')
let ctxMasked = canvasMasked.getContext('2d');


let maskHeight = 240;
let maskWidth = 320;

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
    data: {'img_array': JSON.stringify(cameraShots)},
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
  var img = new Image
  img.onload = function() {
    ctxMasked.drawImage(img, 0, 0, maskWidth, maskHeight)
  }
  img.src = response.resized
  scrollToBottom(1000, textWindow);
}

function snapshot() {
  if (video.videoWidth == 0) {
    return;
  }
  if (canvas.width != video.videoWidth) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvasMaskedWindow.style.border = "1px solid gray"
  }
  if (mediaStream) {
    if (cameraShots.length >= 5) {
      classifyLetters()
      cameraShots = new Array()
    }
    ctx.drawImage(video, 0, 0);
    cameraShots.push(canvas.toDataURL('image/png'));
  }
}

canvasMasked.width = maskWidth
canvasMasked.height = maskHeight
canvas.width = maskWidth
canvas.height = maskHeight
setInterval(snapshot, 200) // 5 frames per second
