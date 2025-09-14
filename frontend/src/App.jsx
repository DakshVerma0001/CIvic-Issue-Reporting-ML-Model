import React, {useState, useRef} from 'react'

// Single-file React component (default export) that uploads an image to the /infer API
// and displays the results (predicted label, confidence, severity, department, raw JSON).
// Tailwind classes are used — include Tailwind in your project or minimal CSS fallback.

export default function App(){
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileRef = useRef()

  const handleFile = (f)=>{
    setError(null)
    setResult(null)
    setFile(f)
    if(!f) { setPreview(null); return }
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  async function submit(){
    if(!file){ setError('Please select an image first'); return }
    setLoading(true)
    setError(null)
    setResult(null)
    try{
      const form = new FormData()
      form.append('image', file)
      // optional extras
      form.append('text','')
      form.append('lat','')
      form.append('lon','')

      const res = await fetch('http://127.0.0.1:5000/infer', { method: 'POST', body: form })
      if(!res.ok){
        const txt = await res.text()
        setError(`Server ${res.status}: ${txt}`)
      } else {
        const data = await res.json()
        setResult(data)
      }
    }catch(e){
      setError(String(e))
    }finally{ setLoading(false) }
  }

  function resetAll(){
    setFile(null); setPreview(null); setResult(null); setError(null)
    if(fileRef.current) fileRef.current.value = null
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center py-12 px-4">
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow p-6">
        <h1 className="text-2xl font-semibold mb-3">Civic Issue Demo — Image → Class + Severity</h1>
        <p className="text-sm text-gray-500 mb-6">Select an image from the pilot dataset or your phone and press <em>Send</em>. The demo calls <code>/infer</code> on the locally running model server (http://127.0.0.1:5000).</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="col-span-1 md:col-span-1">
            <label className="block text-sm font-medium text-gray-700">Image</label>
            <div className="mt-2">
              <input ref={fileRef} accept="image/*" type="file" onChange={e=>handleFile(e.target.files && e.target.files[0])} />
            </div>
            <div className="mt-4 space-x-2">
              <button onClick={submit} disabled={loading} className="px-4 py-2 bg-sky-600 text-white rounded hover:bg-sky-700 disabled:opacity-50">Send</button>
              <button onClick={resetAll} className="px-4 py-2 bg-gray-100 rounded">Reset</button>
            </div>

            {error && <div className="mt-4 text-red-600">{error}</div>}

            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-700">Status</h3>
              <div className="mt-2 text-sm text-gray-600">{loading ? 'Sending…' : (result ? 'Response received' : 'Idle')}</div>
            </div>
          </div>

          <div className="col-span-1 md:col-span-2">
            <div className="flex gap-4">
              <div className="w-1/2">
                <h3 className="text-sm font-medium">Preview</h3>
                <div className="mt-2 border rounded p-2 h-56 flex items-center justify-center bg-gray-50">
                  {preview ? <img src={preview} alt="preview" className="max-h-52 object-contain"/> : <div className="text-gray-400">No image selected</div>}
                </div>
              </div>

              <div className="w-1/2">
                <h3 className="text-sm font-medium">Result</h3>
                <div className="mt-2 border rounded p-2 h-56 overflow-auto bg-gray-50">
                  {loading && <div className="text-gray-500">waiting for server…</div>}
                  {!loading && result && (
                    <div className="space-y-2 text-sm">
                      <div><strong>Predicted:</strong></div>
                      <ol className="pl-4 list-decimal">
                        {Array.isArray(result.predicted) ? result.predicted.map((p,i)=>(
                          <li key={i}>{p[0]} — {Math.round(p[1]*10000)/100}%</li>
                        )) : <li>{JSON.stringify(result.predicted)}</li>}
                      </ol>

                      <div className="pt-2"><strong>Severity:</strong> <span className="font-semibold">{result.severity || '—'}</span></div>
                      <div className="pt-1 text-xs text-gray-500">Department: {result.department || '—'}</div>

                      <details className="mt-2 text-xs">
                        <summary className="cursor-pointer text-sky-600">Show raw response</summary>
                        <pre className="whitespace-pre-wrap max-h-40 overflow-auto text-xs mt-2">{JSON.stringify(result, null, 2)}</pre>
                      </details>
                    </div>
                  )}

                  {!loading && !result && <div className="text-xs text-gray-400">No result yet</div>}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 text-sm text-gray-500">Tip: run the server locally with <code>python -m deploy.inference_api</code> and keep it running while using the UI.</div>
      </div>
    </div>
  )
}

