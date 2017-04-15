const electron = require('electron')
const {app, BrowserWindow} = electron
const {ipcMain} = electron
const {dialog} = electron
const path = require('path')
const url = require('url')


let mainWindow = null
let askForExitWindow = null
let helpWindow = null
let conversationWindow = null
let supby = null
let apiUrl = 'http://localhost:5000'

function createHelpWindow() {
  if (helpWindow === null) {
    helpWindow = new BrowserWindow({
      frame: false,
      parent: mainWindow,
      modal: true,
      show: false,
      width: 400,
      height: 600,
      toolbar: false,
      skipToolkit: true,
      resizable: false
    })
  }
  helpWindow.loadURL(url.format({
    pathname: path.join(__dirname, 'help.html'),
    protocol: 'file:',
    slashes: true
  }))
  helpWindow.setMenu(null)
  helpWindow.show()
}

function createConversationWindow() {
  conversationWindow = new BrowserWindow({
    backgroundColor: '#000000'
  })
  conversationWindow.setMenu(null)
  conversationWindow.setFullScreen(true)

  conversationWindow.loadURL(url.format({
    pathname: path.join(__dirname, 'conversation.html'),
    protocol: 'file:',
    slashes: true
  }))

  conversationWindow.on('closed', () => {
    conversationWindow = null
  })

  conversationWindow.webContents.openDevTools()
}

function createExitWindow() {

  if (askForExitWindow === null) {
    askForExitWindow = new BrowserWindow({
      frame: false,
      parent: mainWindow,
      modal: true,
      show: false,
      width: 400,
      height: 200,
      toolbar: false,
      skipToolkit: true,
      resizable: false,
    })
  }
  askForExitWindow.loadURL(url.format({
    pathname: path.join(__dirname, 'exit.html'),
    protocol: 'file:',
    slashes: true
  }))
  askForExitWindow.setMenu(null)
  askForExitWindow.show()
}

function createWindow () {
  mainWindow = new BrowserWindow({
    backgroundColor: '#c8c8c8'
  })
  mainWindow.setMenu(null)
  mainWindow.setFullScreen(true)

  mainWindow.loadURL(url.format({
    pathname: path.join(__dirname, 'index.html'),
    protocol: 'file:',
    slashes: true
  }))

  mainWindow.on('closed', () => {
    mainWindow = null
  })
  mainWindow.webContents.openDevTools()
}

app.on('ready', () => {
  subpy = require('child_process').spawn('python', [__dirname + '/api/main.py'])
  var startUp = function() {
    require('request-promise')(apiUrl).then(function() {
      createWindow();
    }).catch(function(err) {
      startUp();
    })
  }
  startUp();
  // createWindow();
})

app.on('quit', () => {
  if (subpy !== null) {
    subpy.kill('SIGINT')
  }
})

ipcMain.on('ask-for-exit', (event, args) => {
  createExitWindow()
})

ipcMain.on('exit', (event, args) => {
  app.quit()
})

ipcMain.on('stay', (event, args) => {
  if (askForExitWindow !== null) {
    askForExitWindow.hide()
    askForExitWindow.close()
    askForExitWindow = null
  }
  if (helpWindow !== null) {
    helpWindow.hide()
    helpWindow.close()
    helpWindow = null
  }
})

ipcMain.on('ask-for-help', (event, args) => {
  createHelpWindow()
})

ipcMain.on('ask-for-help-leave', (event, args) => {
  helpWindow.hide()
  helpWindow.close()
  helpWindow = null
})

ipcMain.on('begin-conversation', (event, args) => {
  createConversationWindow()
  mainWindow.close()
})
