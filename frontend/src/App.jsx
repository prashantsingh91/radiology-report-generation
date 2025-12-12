import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// Use environment variable or default to current host
const API_URL = import.meta.env.VITE_API_URL || (window.location.protocol + '//' + window.location.hostname + ':8004')

function App() {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [healthStatus, setHealthStatus] = useState(null)
  
  // DICOM states
  const [dicomFile, setDicomFile] = useState(null)
  const [dicomImage, setDicomImage] = useState(null)
  const [dicomMetadata, setDicomMetadata] = useState(null)
  const [dicomLoading, setDicomLoading] = useState(false)
  const [dicomError, setDicomError] = useState('')
  const [srFindings, setSrFindings] = useState('')
  const [srLoading, setSrLoading] = useState(false)
  const [srError, setSrError] = useState('')
  
  // Report generation states


  const [selectedSections, setSelectedSections] = useState(['FINDINGS', 'IMPRESSION'])
  const [userFindings, setUserFindings] = useState('')
  const [generatedReport, setGeneratedReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')
  
  // Preview states
  const [previewHtml, setPreviewHtml] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  
  // Debug states for raw prompt/output
  const [showRawPrompt, setShowRawPrompt] = useState(false)
  const [showRawOutput, setShowRawOutput] = useState(false)
  
  const [activeTab, setActiveTab] = useState('prompt') // 'prompt', 'dicom', 'report', or 'openai'
  
  // OpenAI report generation states
  const [openaiFindings, setOpenaiFindings] = useState('')
  const [openaiTemplate, setOpenaiTemplate] = useState('')
  const [openaiTemplateFile, setOpenaiTemplateFile] = useState(null)
  const [openaiReport, setOpenaiReport] = useState(null)
  const [openaiLoading, setOpenaiLoading] = useState(false)
  const [openaiError, setOpenaiError] = useState('')
  const [openaiModel, setOpenaiModel] = useState('gpt-5.1')
  // OpenAI patient metadata and routing states
  const [openaiModality, setOpenaiModality] = useState('XRAY') // MRI, XRAY, CT, Ultrasound
  const [openaiBodyPart, setOpenaiBodyPart] = useState('chest') // brain, chest, abdomen, pelvis, spine, extremity, cardiac, neck, general
  const [openaiSex, setOpenaiSex] = useState('')
  const [openaiAge, setOpenaiAge] = useState('')
  const [openaiViewPosition, setOpenaiViewPosition] = useState('')
  const [openaiSpecialty, setOpenaiSpecialty] = useState('auto') // auto or specific override

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`)
      setHealthStatus(response.data)
    } catch (err) {
      setHealthStatus({ status: 'error', message: err.message })
    }
  }

  // Check health on component mount
  useEffect(() => {
    checkHealth()
    // Refresh health status every 10 seconds
    const interval = setInterval(() => {
      checkHealth()
    }, 10000)
    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // No template loading required for generate-report-custom

  // No preview loading for generate-report-custom

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    setLoading(true)
    setError('')
    setOutput('')

    try {
      const response = await axios.post(`${API_URL}/generate`, {
        prompt: prompt,
        max_length: 512
      })
      setOutput(response.data.output)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleDicomUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.dcm') && !file.name.toLowerCase().endsWith('.dicom')) {
      setDicomError('Please upload a DICOM file (.dcm or .dicom)')
      return
    }

    setDicomFile(file)
    setDicomLoading(true)
    setDicomError('')
    setDicomImage(null)
    setDicomMetadata(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_URL}/upload-dicom`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      setDicomImage(response.data.image_base64)
      setDicomMetadata(response.data.metadata)
      setActiveTab('dicom')
    } catch (err) {
      setDicomError(err.response?.data?.detail || err.message || 'Error uploading DICOM file')
      console.error('Error:', err)
    } finally {
      setDicomLoading(false)
    }
  }

  const categorizeMetadata = (metadata) => {
    const categories = {
      patient: [],
      study: [],
      image: [],
      technical: []
    }

    Object.entries(metadata).forEach(([key, value]) => {
      const lowerKey = key.toLowerCase()
      if (lowerKey.includes('patient') || lowerKey.includes('name') || lowerKey.includes('id')) {
        categories.patient.push({ key, value })
      } else if (lowerKey.includes('study') || lowerKey.includes('exam') || lowerKey.includes('series')) {
        categories.study.push({ key, value })
      } else if (lowerKey.includes('image') || lowerKey.includes('pixel') || lowerKey.includes('size') || lowerKey.includes('bits')) {
        categories.image.push({ key, value })
      } else {
        categories.technical.push({ key, value })
      }
    })

    return categories
  }

  const handleGenerateReport = async () => {
    if (!userFindings.trim()) {
      setReportError('Please enter radiologist findings')
      return
    }

    setReportLoading(true)
    setReportError('')
    setGeneratedReport(null)

    try {
      // Extract patient metadata from DICOM if available
      const patientMetadata = dicomMetadata ? {
        PATIENT_NAME: dicomMetadata['Patient\'s Name'] || dicomMetadata['Patient Name'] || 'N/A',
        PATIENT_ID: dicomMetadata['Patient ID'] || dicomMetadata['Patient\'s ID'] || 'N/A',
        PATIENT_AGE: dicomMetadata['Patient\'s Age'] || dicomMetadata['Patient Age'] || 'N/A',
        PATIENT_SEX: dicomMetadata['Patient\'s Sex'] || dicomMetadata['Patient Sex'] || 'N/A',
        STUDY_DATE: dicomMetadata['Study Date'] || dicomMetadata['StudyDate'] || new Date().toISOString().split('T')[0],
        MODALITY: dicomMetadata['Modality'] || 'X-Ray',
        REFERRING_PHYSICIAN: dicomMetadata['Referring Physician\'s Name'] || dicomMetadata['Referring Physician'] || 'N/A',
        BODY_PART: dicomMetadata['Body Part Examined'] || dicomMetadata['Body Part'] || dicomMetadata['BodyPartExamined'] || 'CHEST',
        VIEW_POSITION: dicomMetadata['View Position'] || dicomMetadata['ViewPosition'] || dicomMetadata['Patient Position'] || null
      } : null

      const response = await axios.post(`${API_URL}/generate-report-custom`, {
        template_parameters: selectedSections,
        user_findings: userFindings,
        // region required by backend schema; derive from BODY_PART if present, else 'general'
        region: (patientMetadata && patientMetadata.BODY_PART ? String(patientMetadata.BODY_PART).toLowerCase() : 'general'),
        patient_metadata: patientMetadata
      })

      setGeneratedReport(response.data)
    } catch (err) {
      setReportError(err.response?.data?.detail || err.message || 'Error generating report')
      console.error('Error:', err)
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏥 MedGemma Medical AI Assistant</h1>
        <p>Enter your medical prompt below and get AI-generated responses</p>
      </header>

      <div className="container">
        {/* Device Status Indicator */}
        {healthStatus && healthStatus.model_loaded && (
          <div className={`device-indicator ${healthStatus.gpu_available ? 'gpu' : 'cpu'}`}>
            <div className="device-icon">
              {healthStatus.gpu_available ? '🎮' : '💻'}
            </div>
            <div className="device-info">
              <div className="device-type">
                {healthStatus.gpu_available ? 'GPU' : 'CPU'} Mode
              </div>
              {healthStatus.gpu_name && (
                <div className="device-name">{healthStatus.gpu_name}</div>
              )}
              <div className="device-detail">
                Device: {healthStatus.model_device || 'Unknown'}
              </div>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={`tab-btn ${activeTab === 'prompt' ? 'active' : ''}`}
            onClick={() => setActiveTab('prompt')}
          >
            💬 Text Prompt
          </button>
          <button 
            className={`tab-btn ${activeTab === 'dicom' ? 'active' : ''}`}
            onClick={() => setActiveTab('dicom')}
          >
            🏥 DICOM Viewer
          </button>
          <button 
            className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`}
            onClick={() => setActiveTab('report')}
          >
            📋 Report Generator
          </button>
          <button 
            className={`tab-btn ${activeTab === 'openai' ? 'active' : ''}`}
            onClick={() => setActiveTab('openai')}
          >
            🤖 OpenAI Report
          </button>
        </div>

        <div className="health-check">
          <button onClick={checkHealth} className="health-btn">
            Refresh Status
          </button>
          {healthStatus && (
            <div className={`health-status ${healthStatus.status === 'healthy' ? 'healthy' : 'error'}`}>
              <p><strong>Status:</strong> {healthStatus.status}</p>
              <p><strong>Model Loaded:</strong> {healthStatus.model_loaded ? '✅ Yes' : '❌ No'}</p>
              {healthStatus.gpu_available !== undefined && (
                <p><strong>GPU Available:</strong> {healthStatus.gpu_available ? '✅ Yes' : '❌ No'}</p>
              )}
            </div>
          )}
        </div>

        {/* Prompt Tab */}
        {activeTab === 'prompt' && (
          <>
            <form onSubmit={handleSubmit} className="prompt-form">
              <div className="form-group">
                <label htmlFor="prompt">Enter your prompt:</label>
                <textarea
                  id="prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g., What are the symptoms of pneumonia?"
                  rows={6}
                  disabled={loading}
                />
              </div>

              <button 
                type="submit" 
                className="submit-btn"
                disabled={loading || !prompt.trim()}
              >
                {loading ? 'Generating...' : 'Generate Response'}
              </button>
            </form>

            {error && (
              <div className="error-message">
                <strong>Error:</strong> {error}
              </div>
            )}

            {output && (
              <div className="output-section">
                <h2>Generated Response:</h2>
                <div className="output-content">
                  {output}
                </div>
              </div>
            )}
          </>
        )}

        {/* DICOM Tab */}
        {activeTab === 'dicom' && (
          <div className="dicom-section">
            <div className="form-group">
              <label htmlFor="dicom-upload" className="file-upload-label">
                📁 Upload DICOM File (.dcm or .dicom)
              </label>
              <input
                id="dicom-upload"
                type="file"
                accept=".dcm,.dicom"
                onChange={handleDicomUpload}
                className="file-input"
                disabled={dicomLoading}
              />
              {dicomFile && (
                <p className="file-name">Selected: {dicomFile.name}</p>
              )}
            </div>

            {dicomLoading && (
              <div className="loading-message">Processing DICOM file...</div>
            )}

            {dicomError && (
              <div className="error-message">
                <strong>Error:</strong> {dicomError}
              </div>
            )}

            {dicomImage && dicomMetadata && (
              <div className="dicom-viewer">
                <div className="dicom-layout">
                  <div className="dicom-image-section">
                    <h3>🖼️ DICOM Image</h3>
                    <img 
                      src={dicomImage} 
                      alt="DICOM" 
                      className="dicom-image"
                    />
                    {/* SR Findings trigger and display (if SR metadata present) */}
                    <div style={{ marginTop: '12px' }}>
                      <button
                        className="submit-btn"
                        disabled={srLoading}
                        onClick={async () => {
                          try {
                            setSrLoading(true)
                            setSrError('')
                            setSrFindings('')
                            // Send full metadata; backend will use nested Content Sequence if present
                            const response = await axios.post(`${API_URL}/interpret-sr`, {
                              sr_json: dicomMetadata,
                              model: 'gpt-5.1'
                            })
                            setSrFindings(response.data.findings || '')
                          } catch (err) {
                            const errorMsg = err.response?.data?.detail || err.message || 'Error interpreting SR'
                            setSrError(errorMsg)
                          } finally {
                            setSrLoading(false)
                          }
                        }}
                      >
                        {srLoading ? 'Getting Findings...' : 'Get Finding'}
                      </button>
                    </div>
                    {(srError || srFindings) && (
                      <div style={{ marginTop: '12px' }}>
                        {srError && (
                          <div className="error-message">
                            <strong>Error:</strong> {srError}
                          </div>
                        )}
                        {srFindings && (
                          <div className="report-panel" style={{ marginTop: '10px' }}>
                            <div className="report-panel-header">
                              <h3>📝 Findings (SR)</h3>
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(srFindings)
                                  alert('Findings copied to clipboard!')
                                }}
                                className="copy-btn-small"
                              >
                                📋 Copy
                              </button>
                            </div>
                            <div className="report-content">
                              <pre className="report-pre">{srFindings}</pre>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="dicom-metadata-section">
                    <h3>📋 Metadata ({Object.keys(dicomMetadata).length} fields)</h3>
                    <div className="metadata-categories">
                      {(() => {
                        const categories = categorizeMetadata(dicomMetadata)
                        return (
                          <>
                            {categories.patient.length > 0 && (
                              <div className="metadata-category">
                                <h4>👤 Patient Information</h4>
                                <div className="metadata-list">
                                  {categories.patient.map(({ key, value }) => (
                                    <div key={key} className="metadata-item">
                                      <strong>{key}:</strong> <span>{String(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {categories.study.length > 0 && (
                              <div className="metadata-category">
                                <h4>🏥 Study Information</h4>
                                <div className="metadata-list">
                                  {categories.study.map(({ key, value }) => (
                                    <div key={key} className="metadata-item">
                                      <strong>{key}:</strong> <span>{String(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {categories.image.length > 0 && (
                              <div className="metadata-category">
                                <h4>🖼️ Image Properties</h4>
                                <div className="metadata-list">
                                  {categories.image.map(({ key, value }) => (
                                    <div key={key} className="metadata-item">
                                      <strong>{key}:</strong> <span>{String(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {categories.technical.length > 0 && (
                              <div className="metadata-category">
                                <h4>🔧 Technical Details</h4>
                                <div className="metadata-list">
                                  {categories.technical.map(({ key, value }) => (
                                    <div key={key} className="metadata-item">
                                      <strong>{key}:</strong> <span>{String(value)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </>
                        )
                      })()}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Report Generator Tab */}
        {activeTab === 'report' && (
          <div className="report-section">
            <div className="report-config">
              {/* Sections to generate */}
              <div className="form-group">
                <label>Sections to generate (order preserved):</label>
                <div className="checkbox-group">
                  {['CLINICAL_DETAILS', 'FINDINGS', 'IMPRESSION', 'RECOMMENDATIONS'].map(section => (
                    <label key={section} className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={selectedSections.includes(section)}
                        onChange={(e) => {
                          const checked = e.target.checked
                          setSelectedSections(prev => {
                            if (checked) {
                              // Add and keep default order as defined above
                              const order = ['CLINICAL_DETAILS','FINDINGS','IMPRESSION','RECOMMENDATIONS']
                              const next = Array.from(new Set([...prev, section]))
                              return next.sort((a,b) => order.indexOf(a) - order.indexOf(b))
                            }
                            return prev.filter(s => s !== section)
                          })
                        }}
                        disabled={reportLoading}
                      />
                      <span style={{ marginLeft: 8 }}>{section.replace('_',' ')}</span>
                    </label>
                  ))}
              </div>
                <p className="input-hint">Tip: Common choice is FINDINGS + IMPRESSION.</p>
              </div>

              {dicomMetadata && (
                <div className="dicom-info-banner">
                  ✅ Using patient metadata from uploaded DICOM file
                </div>
              )}
            </div>

            {/* No template preview required for generate-report-custom */}

            <div className="form-group">
              <label htmlFor="findings">Radiologist Findings/Observations:</label>
              <textarea
                id="findings"
                value={userFindings}
                onChange={(e) => setUserFindings(e.target.value)}
                placeholder="Enter findings (e.g., 'bilateral lower lobe opacities, cardiomegaly, no pneumothorax') or detailed multi-line observations"
                rows={8}
                disabled={reportLoading}
                className="findings-input"
              />
              <p className="input-hint">
                Enter raw findings from the radiologist. Can be one word, a phrase, or multiple lines of detailed observations.
              </p>
            </div>

            <button
              onClick={handleGenerateReport}
              className="submit-btn"
              disabled={reportLoading || !userFindings.trim() || selectedSections.length === 0}
            >
              {reportLoading ? 'Generating Report...' : 'Generate Report'}
            </button>

            {reportError && (
              <div className="error-message">
                <strong>Error:</strong> {reportError}
              </div>
            )}

            {generatedReport && (
              <div className="report-viewer">
                <h2>Generated Radiology Report</h2>
                  {/* MedGemma Report */}
                  <div className="report-panel">
                    <div className="report-panel-header">
                      <h3>🤖 MedGemma Report</h3>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(generatedReport.full_report)
                          alert('MedGemma report copied to clipboard!')
                        }}
                        className="copy-btn-small"
                      >
                        📋 Copy
                      </button>
                    </div>
                    <div className="report-content">
                      <pre className="report-pre">{generatedReport.full_report}</pre>
                    </div>
                </div>
                
                {/* Debug Section: Raw Prompt and Output */}
                <div className="debug-section">
                  <h3>🔍 Debug Information</h3>
                  
                  {generatedReport.raw_prompt && (
                    <div className="debug-item">
                      <button
                        onClick={() => setShowRawPrompt(!showRawPrompt)}
                        className="debug-toggle-btn"
                      >
                        {showRawPrompt ? '▼' : '▶'} Raw Prompt Sent to Model
                      </button>
                      {showRawPrompt && (
                        <div className="debug-content">
                          <pre className="debug-pre">{generatedReport.raw_prompt}</pre>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(generatedReport.raw_prompt)
                              alert('Prompt copied to clipboard!')
                            }}
                            className="copy-btn-small"
                          >
                            📋 Copy Prompt
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {generatedReport.raw_output && (
                    <div className="debug-item">
                      <button
                        onClick={() => setShowRawOutput(!showRawOutput)}
                        className="debug-toggle-btn"
                      >
                        {showRawOutput ? '▼' : '▶'} Raw Model Output (Before Post-Processing)
                      </button>
                      {showRawOutput && (
                        <div className="debug-content">
                          <pre className="debug-pre">{generatedReport.raw_output}</pre>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(generatedReport.raw_output)
                              alert('Raw output copied to clipboard!')
                            }}
                            className="copy-btn-small"
                          >
                            📋 Copy Raw Output
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* OpenAI Report Generator Tab */}
        {activeTab === 'openai' && (
          <div className="report-section">
            <div className="report-config">
              <h2>🤖 OpenAI Report Generator</h2>
              <p className="input-hint">Generate radiology reports using OpenAI's latest models. Enter findings and provide a template. The prompt comes from prompts/template_filling_prompt.txt.</p>

              <div className="form-group">
                <label htmlFor="openai-model">Select OpenAI Model:</label>
                <select
                  id="openai-model"
                  value={openaiModel}
                  onChange={(e) => setOpenaiModel(e.target.value)}
                  className="region-select"
                >
                  <option value="gpt-5.1">GPT-5.1 (Latest)</option>
                  <option value="gpt-5">GPT-5</option>
                  <option value="gpt-4o">GPT-4o</option>
                  <option value="gpt-4-turbo">GPT-4 Turbo</option>
                  <option value="gpt-4">GPT-4</option>
                  <option value="o1-preview">O1 Preview (Reasoning)</option>
                  <option value="o1-mini">O1 Mini (Reasoning)</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="openai-findings">Radiologist Findings/Observations:</label>
                <textarea
                  id="openai-findings"
                  value={openaiFindings}
                  onChange={(e) => setOpenaiFindings(e.target.value)}
                  placeholder="Enter findings (e.g., 'bilateral lower lobe opacities, cardiomegaly')"
                  rows={6}
                  disabled={openaiLoading}
                  className="findings-input"
                />
              </div>

              {/* Modality and Body Part selection */}
              <div className="form-row">
                <div className="form-group" style={{ minWidth: '240px' }}>
                  <label htmlFor="openai-modality">Modality (supported only):</label>
                  <select
                    id="openai-modality"
                    value={openaiModality}
                    onChange={(e) => setOpenaiModality(e.target.value)}
                    className="region-select"
                    disabled={openaiLoading}
                  >
                    <option value="MRI">MRI</option>
                    <option value="XRAY">XRAY</option>
                    <option value="CT">CT</option>
                    <option value="ULTRASOUND">Ultrasound</option>
                  </select>
                  <p className="input-hint">Only MRI, XRAY, CT, Ultrasound are supported.</p>
                </div>
                <div className="form-group" style={{ minWidth: '240px' }}>
                  <label htmlFor="openai-bodypart">Body Part:</label>
                  <select
                    id="openai-bodypart"
                    value={openaiBodyPart}
                    onChange={(e) => setOpenaiBodyPart(e.target.value)}
                    className="region-select"
                    disabled={openaiLoading}
                  >
                    <option value="brain">Brain</option>
                    <option value="chest">Chest</option>
                    <option value="abdomen">Abdomen</option>
                    <option value="pelvis">Pelvis</option>
                    <option value="spine">Spine</option>
                    <option value="extremity">Extremity</option>
                    <option value="cardiac">Cardiac</option>
                    <option value="neck">Neck</option>
                    <option value="general">General</option>
                  </select>
                </div>
              </div>

              {/* Patient metadata inputs */}
              <div className="form-row">
                <div className="form-group" style={{ minWidth: '160px' }}>
                  <label htmlFor="openai-sex">Patient Sex:</label>
                  <input
                    id="openai-sex"
                    type="text"
                    value={openaiSex}
                    onChange={(e) => setOpenaiSex(e.target.value)}
                    placeholder="M / F / O"
                    disabled={openaiLoading}
                  />
                </div>
                <div className="form-group" style={{ minWidth: '160px' }}>
                  <label htmlFor="openai-age">Patient Age:</label>
                  <input
                    id="openai-age"
                    type="text"
                    value={openaiAge}
                    onChange={(e) => setOpenaiAge(e.target.value)}
                    placeholder="e.g., 45"
                    disabled={openaiLoading}
                  />
                </div>
                <div className="form-group" style={{ minWidth: '200px' }}>
                  <label htmlFor="openai-viewpos">View Position (optional):</label>
                  <input
                    id="openai-viewpos"
                    type="text"
                    value={openaiViewPosition}
                    onChange={(e) => setOpenaiViewPosition(e.target.value)}
                    placeholder="e.g., AP, PA, LAT"
                    disabled={openaiLoading}
                  />
                </div>
              </div>

              {/* Specialty override */}
              <div className="form-group" style={{ minWidth: '240px' }}>
                <label htmlFor="openai-specialty">Specialty (override):</label>
                <select
                  id="openai-specialty"
                  value={openaiSpecialty}
                  onChange={(e) => setOpenaiSpecialty(e.target.value)}
                  className="region-select"
                  disabled={openaiLoading}
                >
                  <option value="auto">Auto-detect (recommended)</option>
                  <option value="neuro-radiology">Neuro-radiology</option>
                  <option value="chest radiology">Chest radiology</option>
                  <option value="MSK radiology">MSK radiology</option>
                  <option value="abdominal radiology">Abdominal radiology</option>
                  <option value="cardiac imaging">Cardiac imaging</option>
                  <option value="ultrasound imaging">Ultrasound imaging</option>
                  <option value="medical imaging">General medical imaging</option>
                </select>
                <p className="input-hint">If set, overrides auto-detected specialty.</p>
              </div>

              <div className="form-group">
                <label htmlFor="openai-template">Template (JSON or Text):</label>
                <div style={{ marginBottom: '10px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  <label htmlFor="openai-template-upload" className="file-upload-label">
                    📄 Upload Template File
                  </label>
                </div>
                <input
                  id="openai-template-upload"
                  type="file"
                  accept=".json,.txt"
                  onChange={async (e) => {
                    const file = e.target.files[0]
                    if (file) {
                      setOpenaiTemplateFile(file)
                      const text = await file.text()
                      setOpenaiTemplate(text)
                    }
                  }}
                  className="file-input"
                  disabled={openaiLoading}
                  style={{ marginBottom: '10px' }}
                />
                {openaiTemplateFile && (
                  <p className="file-name">Selected: {openaiTemplateFile.name}</p>
                )}
                <textarea
                  id="openai-template"
                  value={openaiTemplate}
                  onChange={(e) => setOpenaiTemplate(e.target.value)}
                  placeholder="Enter template text or JSON. The prompt will come from prompts/template_filling_prompt.txt"
                  rows={12}
                  disabled={openaiLoading}
                  className="findings-input"
                />
                <p className="input-hint">
                  Enter the template structure (e.g., section headers and format). The prompt instructions come from prompts/template_filling_prompt.txt automatically.
                </p>
              </div>

              <button
                onClick={async () => {
                  if (!openaiFindings.trim()) {
                    setOpenaiError('Please enter findings')
                    return
                  }
                  if (!openaiTemplate.trim()) {
                    setOpenaiError('Please enter or upload a template')
                    return
                  }

                  setOpenaiLoading(true)
                  setOpenaiError('')
                  setOpenaiReport(null)

                  try {
                    const formData = new FormData()
                    formData.append('findings', openaiFindings)
                    formData.append('template', openaiTemplate)
                    formData.append('model', openaiModel)

                    // Build patient metadata using UI inputs, falling back to DICOM
                      const patientMetadata = {
                      PATIENT_SEX: (openaiSex || (dicomMetadata && (dicomMetadata['Patient\'s Sex'] || dicomMetadata['Patient Sex']))) || 'N/A',
                      PATIENT_AGE: (openaiAge || (dicomMetadata && (dicomMetadata['Patient\'s Age'] || dicomMetadata['Patient Age']))) || 'N/A',
                      BODY_PART: openaiBodyPart ? openaiBodyPart.toUpperCase() : ((dicomMetadata && (dicomMetadata['Body Part Examined'] || dicomMetadata['Body Part'])) || 'GENERAL'),
                      MODALITY: openaiModality ? openaiModality.toUpperCase() : ((dicomMetadata && (dicomMetadata['Modality'])) || 'XRAY'),
                      VIEW_POSITION: openaiViewPosition || (dicomMetadata && (dicomMetadata['View Position'] || dicomMetadata['ViewPosition'])) || null
                      }
                      formData.append('patient_metadata', JSON.stringify(patientMetadata))

                    // Add specialty override if not auto
                    if (openaiSpecialty && openaiSpecialty !== 'auto') {
                      formData.append('specialty', openaiSpecialty)
                    }

                    const response = await axios.post(`${API_URL}/fill-template`, formData, {
                      headers: {
                        'Content-Type': 'multipart/form-data'
                      }
                    })

                    setOpenaiReport(response.data)
                  } catch (err) {
                    const errorMsg = err.response?.data?.detail || err.response?.data?.message || err.message || 'Error generating OpenAI report'
                    const statusCode = err.response?.status
                    const fullError = statusCode ? `[${statusCode}] ${errorMsg}` : errorMsg
                    setOpenaiError(fullError)
                    console.error('Error details:', {
                      message: err.message,
                      status: err.response?.status,
                      statusText: err.response?.statusText,
                      data: err.response?.data,
                      url: `${API_URL}/fill-template`
                    })
                  } finally {
                    setOpenaiLoading(false)
                  }
                }}
                className="submit-btn"
                disabled={openaiLoading || !openaiFindings.trim() || !openaiTemplate.trim()}
              >
                {openaiLoading ? 'Generating Report...' : 'Generate OpenAI Report'}
              </button>

              {openaiError && (
                <div className="error-message">
                  <strong>Error:</strong> {openaiError}
                </div>
              )}

              {openaiReport && (
                <div className="report-viewer">
                  <h2>Generated Report</h2>
                  <div className="report-panel">
                    <div className="report-panel-header">
                      <h3>🤖 OpenAI Report ({openaiReport.openai_model || openaiReport.model_used})</h3>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(openaiReport.report)
                          alert('Report copied to clipboard!')
                        }}
                        className="copy-btn-small"
                      >
                        📋 Copy
                      </button>
                    </div>
                    <div className="report-content">
                      <div className="formatted-report">
                        {openaiReport.report.split('\n').map((line, index) => {
                          // Format section headers
                          if (line.match(/^[A-Z][A-Z\s]+:$/)) {
                            return <h4 key={index} className="report-section-header">{line}</h4>
                          }
                          // Format bold text (if any)
                          if (line.trim().startsWith('**') && line.trim().endsWith('**')) {
                            return <strong key={index} className="report-bold">{line.replace(/\*\*/g, '')}</strong>
                          }
                          // Regular lines
                          if (line.trim()) {
                            return <p key={index} className="report-line">{line}</p>
                          }
                          // Empty lines for spacing
                          return <br key={index} />
                        })}
                      </div>
                      {/* Also show raw text in a collapsible section for copying */}
                      <details style={{ marginTop: '20px' }}>
                        <summary style={{ cursor: 'pointer', color: '#666', fontSize: '14px' }}>
                          View Raw Text
                        </summary>
                        <pre className="report-pre" style={{ marginTop: '10px', fontSize: '12px', maxHeight: '300px', overflow: 'auto' }}>
                          {openaiReport.report}
                        </pre>
                      </details>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

