const form = document.querySelector('#analysis-form')
const input = document.querySelector('#image-input')
const dropZone = document.querySelector('#drop-zone')
const preview = document.querySelector('#file-preview')
const previewImage = document.querySelector('#preview-image')
const previewName = document.querySelector('#preview-name')
const previewSize = document.querySelector('#preview-size')
const removeButton = document.querySelector('#remove-file')
const sampleButton = document.querySelector('#sample-image-button')
const analyzeButton = document.querySelector('#analyze-button')
const emptyResults = document.querySelector('#empty-results')
const loadingResults = document.querySelector('#loading-results')
const analysisResults = document.querySelector('#analysis-results')
const errorResults = document.querySelector('#error-results')
let previewUrl = ''

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function setPanel(panel) {
  emptyResults.hidden = panel !== 'empty'
  loadingResults.hidden = panel !== 'loading'
  analysisResults.hidden = panel !== 'analysis'
  errorResults.hidden = panel !== 'error'
}

function setFile(file) {
  if (!file) return
  if (previewUrl) URL.revokeObjectURL(previewUrl)
  previewUrl = URL.createObjectURL(file)
  previewImage.src = previewUrl
  previewName.textContent = file.name
  previewSize.textContent = `${file.type || 'Unknown type'} · ${formatBytes(file.size)}`
  preview.hidden = false
  dropZone.classList.add('has-file')
  analyzeButton.disabled = false
  setPanel('empty')
}

function clearFile() {
  if (previewUrl) URL.revokeObjectURL(previewUrl)
  previewUrl = ''
  input.value = ''
  preview.hidden = true
  previewImage.removeAttribute('src')
  dropZone.classList.remove('has-file')
  analyzeButton.disabled = true
  setPanel('empty')
}

function createSyntheticSample() {
  const canvas = document.createElement('canvas')
  canvas.width = 640
  canvas.height = 480
  const context = canvas.getContext('2d')
  if (!context) return

  const background = context.createLinearGradient(0, 0, canvas.width, canvas.height)
  background.addColorStop(0, '#dba38f')
  background.addColorStop(1, '#b56f69')
  context.fillStyle = background
  context.fillRect(0, 0, canvas.width, canvas.height)

  const region = context.createRadialGradient(350, 230, 25, 350, 230, 145)
  region.addColorStop(0, '#472326')
  region.addColorStop(0.34, '#7c3c3d')
  region.addColorStop(0.7, '#b9655d')
  region.addColorStop(1, 'rgba(181, 101, 93, 0)')
  context.fillStyle = region
  context.fillRect(170, 50, 360, 360)

  context.strokeStyle = 'rgba(255, 235, 222, 0.22)'
  context.lineWidth = 3
  for (let radius = 45; radius <= 145; radius += 25) {
    context.beginPath()
    context.arc(350, 230, radius, 0, Math.PI * 2)
    context.stroke()
  }

  context.strokeStyle = 'rgba(73, 29, 34, 0.34)'
  context.lineWidth = 2
  for (let index = 0; index < 42; index += 1) {
    const angle = index * 2.399
    const distance = 35 + (index % 9) * 11
    const x = 350 + Math.cos(angle) * distance
    const y = 230 + Math.sin(angle) * distance
    context.beginPath()
    context.moveTo(x - 5, y - 3)
    context.lineTo(x + 6, y + 4)
    context.stroke()
  }

  context.save()
  context.beginPath()
  context.arc(350, 230, 142, 0, Math.PI * 2)
  context.clip()
  context.strokeStyle = 'rgba(63, 29, 35, 0.28)'
  context.lineWidth = 1
  for (let offset = -132; offset <= 132; offset += 12) {
    context.beginPath()
    context.moveTo(218, 230 + offset)
    context.lineTo(482, 230 + offset)
    context.moveTo(350 + offset, 98)
    context.lineTo(350 + offset, 362)
    context.stroke()
  }
  for (let index = 0; index < 420; index += 1) {
    const angle = index * 2.399
    const distance = 12 + (index % 22) * 6
    const x = 350 + Math.cos(angle) * distance
    const y = 230 + Math.sin(angle) * distance
    context.fillStyle = index % 3 === 0
      ? 'rgba(246, 199, 175, 0.7)'
      : 'rgba(53, 24, 31, 0.68)'
    context.beginPath()
    context.arc(x, y, 2 + (index % 3), 0, Math.PI * 2)
    context.fill()
  }
  context.restore()

  canvas.toBlob((blob) => {
    if (!blob) return
    const file = new File([blob], 'synthetic-quality-sample.png', { type: 'image/png' })
    const transfer = new DataTransfer()
    transfer.items.add(file)
    input.files = transfer.files
    setFile(file)
  }, 'image/png')
}

function renderAnalysis(data) {
  const analysis = data.analysis
  const ready = analysis.readiness === 'ready_for_research_pipeline'
  const badge = document.querySelector('#readiness-badge')
  badge.textContent = ready ? 'Ready for research pipeline' : 'Review recommended'
  badge.className = ready ? 'ready' : 'review'
  document.querySelector('#passed-count').textContent = analysis.passed_checks
  document.querySelector('#brightness-metric').textContent = `${analysis.metrics.brightness}%`
  document.querySelector('#contrast-metric').textContent = `${analysis.metrics.contrast}%`
  document.querySelector('#edge-metric').textContent = `${analysis.metrics.edge_detail}%`

  const list = document.querySelector('#check-list')
  list.replaceChildren(
    ...analysis.checks.map((check) => {
      const row = document.createElement('div')
      row.className = check.passed ? 'passed' : 'needs-review'
      const icon = document.createElement('span')
      const content = document.createElement('div')
      const label = document.createElement('strong')
      const detail = document.createElement('small')
      icon.textContent = check.passed ? '✓' : '!'
      label.textContent = check.label
      detail.textContent = check.detail
      content.append(label, detail)
      row.append(icon, content)
      return row
    }),
  )

  const inference = data.inference
  const status = document.querySelector('#inference-status')
  const version = document.querySelector('#model-version')
  const topEstimate = document.querySelector('#top-estimate')
  const predictionList = document.querySelector('#prediction-list')
  const explanation = document.querySelector('#inference-explanation')
  predictionList.replaceChildren()

  if (!inference.performed) {
    status.textContent = 'Inference skipped'
    status.className = 'skipped'
    version.textContent = ''
    topEstimate.hidden = true
    explanation.textContent = inference.reason
    return
  }

  const abstained = inference.status === 'abstained_low_confidence'
  status.textContent = abstained ? 'Low confidence · abstained' : 'Research estimate'
  status.className = abstained ? 'abstained' : 'estimated'
  version.textContent = inference.model_version
  topEstimate.hidden = false
  document.querySelector('#prediction-name').textContent = inference.top_prediction.name
  document.querySelector('#prediction-confidence').textContent =
    `${Math.round(inference.top_prediction.probability * 100)}%`

  predictionList.replaceChildren(
    ...inference.ranked_predictions.map((prediction) => {
      const row = document.createElement('div')
      const heading = document.createElement('div')
      const label = document.createElement('strong')
      const value = document.createElement('span')
      const track = document.createElement('div')
      const bar = document.createElement('i')
      const percentage = Math.round(prediction.probability * 100)
      label.textContent = prediction.name
      value.textContent = `${percentage}%`
      heading.append(label, value)
      track.setAttribute('role', 'progressbar')
      track.setAttribute('aria-label', prediction.name)
      track.setAttribute('aria-valuemin', '0')
      track.setAttribute('aria-valuemax', '100')
      track.setAttribute('aria-valuenow', String(percentage))
      bar.style.width = `${percentage}%`
      track.append(bar)
      row.append(heading, track)
      return row
    }),
  )
  explanation.textContent = abstained
    ? `The highest probability is below the ${(inference.confidence_threshold * 100).toFixed(0)}% validation-derived threshold. No class is accepted.`
    : inference.acquisition_warning
}

input.addEventListener('change', () => setFile(input.files?.[0]))

for (const eventName of ['dragenter', 'dragover']) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault()
    dropZone.classList.add('dragging')
  })
}

for (const eventName of ['dragleave', 'drop']) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault()
    dropZone.classList.remove('dragging')
  })
}

dropZone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  const transfer = new DataTransfer()
  transfer.items.add(file)
  input.files = transfer.files
  setFile(file)
})

removeButton.addEventListener('click', clearFile)
sampleButton.addEventListener('click', createSyntheticSample)

form.addEventListener('submit', async (event) => {
  event.preventDefault()
  const file = input.files?.[0]
  if (!file) return

  setPanel('loading')
  analyzeButton.disabled = true
  const payload = new FormData()
  payload.append('image', file)

  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: payload })
    const data = await response.json()
    if (!response.ok) throw new Error(data.message || 'The image could not be analyzed.')
    renderAnalysis(data)
    setPanel('analysis')
  } catch (error) {
    document.querySelector('#error-message').textContent =
      error instanceof Error ? error.message : 'Choose another image and try again.'
    setPanel('error')
  } finally {
    analyzeButton.disabled = false
  }
})
