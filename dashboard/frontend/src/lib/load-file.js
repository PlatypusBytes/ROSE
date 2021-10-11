export default function getLoadedFileContents(event) {
  return new Promise((resolve) => {
    
    const reader = new FileReader()

    reader.addEventListener('load', (event) => {
      resolve(JSON.parse(event.target.result))
    })
    
    reader.readAsText(event)
  })
}
