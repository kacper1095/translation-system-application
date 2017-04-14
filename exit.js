const {ipcRenderer} = require('electron')

let exitbtn = document.getElementById('btn-leave')
exitbtn.addEventListener('click', function(e) {
  ipcRenderer.send('exit')
})

let staybtn = document.getElementById('btn-stay')
staybtn.addEventListener('click', function(e) {
  ipcRenderer.send('stay')
})
