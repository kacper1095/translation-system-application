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
  createWindow()
})

ipcMain.on('ask-for-exit', (event, args) => {
  createExitWindow()
})

ipcMain.on('exit', (event, args) => {
  askForExitWindow.hide()
  mainWindow.hide()
  app.quit()
})

ipcMain.on('stay', (event, args) => {
  askForExitWindow.hide()
  askForExitWindow.close()
  askForExitWindow = null
})

ipcMain.on('ask-for-help', (event, args) => {
  createHelpWindow()
})

ipcMain.on('ask-for-help-leave', (event, args) => {
  helpWindow.hide()
  helpWindow.close()
  helpWindow = null
})
