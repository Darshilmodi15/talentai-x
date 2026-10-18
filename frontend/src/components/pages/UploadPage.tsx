import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, CheckCircle, XCircle, Loader2, Eye } from 'lucide-react'
import toast from 'react-hot-toast'
import { parseResume, getJobStatus } from '../../api/client'

type JobState = {
  jobId: string
  fileName: string
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'partial'
  candidateId?: string
  parseConfidence?: number
  skillsFound?: number
  error?: string
  file?: File
}

export default function UploadPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState<JobState[]>([])
  const [uploading, setUploading] = useState(false)

  const pollJob = useCallback(async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId)
        setJobs((prev) =>
          prev.map((j) =>
            j.jobId === jobId
              ? {
                  ...j,
                  status: status.status,
                  candidateId: status.candidate_id,
                  parseConfidence: status.parse_confidence,
                  skillsFound: status.skills_found,
                  error: status.error_message,
                }
              : j
          )
        )
        if (['completed', 'failed', 'partial'].includes(status.status)) {
          clearInterval(interval)
          if (status.status === 'completed') {
            toast.success(`"${status.file_name}" parsed successfully`)
          } else if (status.status === 'failed') {
            toast.error(status.error_message || `Failed to parse "${status.file_name}"`)
          }
        }
      } catch (err: any) {
        clearInterval(interval)
        setJobs((prev) =>
          prev.map((j) =>
            j.jobId === jobId
              ? { ...j, status: 'failed', error: err.message || 'Failed to communicate with server' }
              : j
          )
        )
        toast.error(err.message || 'Failed to communicate with server')
      }
    }, 2000)
  }, [])

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setUploading(true)
    for (const file of acceptedFiles) {
      try {
        const result = await parseResume(file)
        const newJob: JobState = {
          jobId: result.job_id,
          fileName: file.name,
          status: 'queued',
          file: file,
        }
        setJobs((prev) => [newJob, ...prev])
        toast.success(`"${file.name}" queued for processing`)
        pollJob(result.job_id)
      } catch (err: any) {
        toast.error(err.message || `Failed to upload ${file.name}`)
      }
    }
    setUploading(false)
  }, [pollJob])

  const retryJob = useCallback((job: JobState) => {
    if (!job.file) return
    setJobs((prev) => prev.filter((j) => j.jobId !== job.jobId))
    onDrop([job.file])
  }, [onDrop])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    maxSize: 10 * 1024 * 1024,
    multiple: true,
  })

  const statusIcon = (status: string) => {
    if (status === 'completed') return <CheckCircle className="w-5 h-5 text-green-500" />
    if (status === 'failed') return <XCircle className="w-5 h-5 text-red-500" />
    return <Loader2 className="w-5 h-5 text-indigo-500 animate-spin" />
  }

  const statusColor = (job: JobState) => {
    if (job.status === 'completed') return 'bg-green-50 border-green-200'
    if (job.status === 'partial') return 'bg-amber-50 border-amber-200'
    if (job.status === 'failed') {
      // Styled warning card for quota/server errors instead of red
      if (job.error?.includes('temporarily unavailable') || job.error?.includes('Server error')) {
        return 'bg-amber-50 border-amber-300'
      }
      return 'bg-red-50 border-red-200'
    }
    return 'bg-indigo-50 border-indigo-200'
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Resume</h1>
        <p className="text-gray-500 mt-1">PDF, DOCX, or TXT • Max 10MB per file</p>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors mb-8 ${
          isDragActive ? 'border-indigo-400 bg-indigo-50' : 'border-gray-300 hover:border-indigo-300 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="w-10 h-10 text-gray-400 mx-auto mb-4" />
        {uploading ? (
          <p className="text-gray-600 font-medium">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-indigo-600 font-medium">Drop files here</p>
        ) : (
          <>
            <p className="text-gray-700 font-medium text-lg">Drag & drop resumes here</p>
            <p className="text-gray-400 text-sm mt-2">or click to browse files</p>
            <p className="text-gray-400 text-xs mt-4">PDF · DOCX · TXT · Multiple files supported</p>
          </>
        )}
      </div>

      {/* Job list */}
      {jobs.length > 0 && (
        <div>
          <h2 className="text-base font-semibold text-gray-900 mb-3">Processing queue</h2>
          <div className="space-y-3">
            {jobs.map((job) => (
              <div
                key={job.jobId}
                className={`border rounded-xl p-4 flex items-center justify-between ${statusColor(job)}`}
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-gray-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{job.fileName}</p>
                    <p className="text-xs text-gray-500 capitalize">
                      {job.status}
                      {job.parseConfidence !== undefined && ` · Confidence: ${Math.round(job.parseConfidence * 100)}%`}
                      {job.skillsFound !== undefined && ` · ${job.skillsFound} skills`}
                    </p>
                    {job.error && (
                      <p className={`text-xs mt-1 font-medium ${job.error.includes('temporarily unavailable') ? 'text-amber-700' : 'text-red-600'}`}>
                        {job.error}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {statusIcon(job.status)}
                  {job.status === 'failed' && job.file && (
                    <button
                      onClick={() => retryJob(job)}
                      className="px-3 py-1.5 text-xs font-medium bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 text-gray-700 transition-colors"
                    >
                      Try Again
                    </button>
                  )}
                  {job.status === 'completed' && job.candidateId && (
                    <button
                      onClick={() => navigate(`/candidates/${job.candidateId}`)}
                      className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 font-medium"
                    >
                      <Eye className="w-3 h-3" />
                      View profile
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
