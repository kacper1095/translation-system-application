const {ipcRenderer} = require('electron')
window.$ = window.jQuery = require('jquery')

let logout = document.getElementById('btn-logout')
let textWindow = document.getElementById('translation')
let canvas = document.querySelector('canvas')
let ctx = canvas.getContext('2d');
let mediaStream = null;
let video = document.getElementById('camera')
let cameraShots = new Array()

logout.addEventListener('click', function(e) {
  ipcRenderer.send('ask-for-exit')
})

navigator.webkitGetUserMedia({video: true},
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
      textWindow.innerHTML = textWindow.innerHTML + response.result;
      scrollToBottom(1000, textWindow);
    },
    error: function(msg) {
      console.log(msg)
    }
  })
}

function snapshot() {
  if (mediaStream) {
    if (cameraShots.length >= 5) {
      classifyLetters()
      cameraShots = new Array()
    }
    ctx.drawImage(video, 0, 0);
    cameraShots.push(canvas.toDataURL('image/png'));
  }
}

setInterval(snapshot, 200) // 5 frames per second
