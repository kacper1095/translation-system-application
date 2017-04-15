const {ipcRenderer} = require('electron')

let staybtn = document.getElementById('btn-stay')
staybtn.addEventListener('click', function(e) {
  ipcRenderer.send('stay')
})
