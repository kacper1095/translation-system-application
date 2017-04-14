const {ipcRenderer} = require('electron')

let logout = document.getElementById('btn-logout')
logout.addEventListener('click', function(e) {
  // let window = remote.getCurrentWindow()
  // window.close()
  // app.send('ask-for-exit')
  ipcRenderer.send('ask-for-exit')
})

let beginConversation = document.getElementById('btn-begin-conversation')
beginConversation.addEventListener('click', function(e) {
  ipcRenderer.send('begin-conversation')
})
