import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Send, Eye, FileText, AlertTriangle } from 'lucide-react';
import {
  fetchSubstackStatus,
  composeSubstackPost,
  createSubstackDraft,
  publishSubstackDraft,
  fetchBacktestRuns,
} from '../../lib/api';
import { SubstackPostContent, SubstackComposeRequest } from '../../types/substack';
import PostPreview from './PostPreview';
import DraftsList from './DraftsList';

const POST_TYPES = [
  { value: 'daily_signals', label: 'Daily Signals' },
  { value: 'backtest_report', label: 'Backtest Report' },
  { value: 'signal_note', label: 'Quick Note' },
  { value: 'custom', label: 'Custom Post' },
];

export default function PublishPage() {
  const [postType, setPostType] = useState<string>('daily_signals');
  const [runId, setRunId] = useState('');
  const [signalId, setSignalId] = useState('');
  const [customTitle, setCustomTitle] = useState('');
  const [customMarkdown, setCustomMarkdown] = useState('');
  const [customTags, setCustomTags] = useState('');
  const [preview, setPreview] = useState<SubstackPostContent | null>(null);
  const [composedDrafts, setComposedDrafts] = useState<SubstackPostContent[]>([]);
  const [selectedDraftIndex, setSelectedDraftIndex] = useState<number | null>(null);
  const [draftId, setDraftId] = useState<number | null>(null);

  const statusQuery = useQuery({
    queryKey: ['substack-status'],
    queryFn: fetchSubstackStatus,
  });

  const runsQuery = useQuery({
    queryKey: ['backtest-runs'],
    queryFn: fetchBacktestRuns,
  });

  const composeMutation = useMutation({
    mutationFn: composeSubstackPost,
    onSuccess: (data) => {
      setPreview(data);
      setComposedDrafts((prev) => [data, ...prev]);
      setSelectedDraftIndex(0);
    },
  });

  const draftMutation = useMutation({
    mutationFn: createSubstackDraft,
    onSuccess: (data) => {
      setDraftId((data.draft as { id: number }).id);
      setPreview(data.post);
    },
  });

  const publishMutation = useMutation({
    mutationFn: ({ id, sendEmail }: { id: number; sendEmail: boolean }) =>
      publishSubstackDraft(id, sendEmail),
  });

  const buildRequest = (): SubstackComposeRequest => {
    const req: SubstackComposeRequest = { post_type: postType as SubstackComposeRequest['post_type'] };
    if (postType === 'backtest_report') req.run_id = runId;
    if (postType === 'signal_note') req.signal_id = signalId;
    if (postType === 'custom') {
      req.title = customTitle;
      req.markdown = customMarkdown;
      req.tags = customTags.split(',').map((t) => t.trim()).filter(Boolean);
    }
    return req;
  };

  const handleCompose = () => composeMutation.mutate(buildRequest());
  const handleCreateDraft = () => draftMutation.mutate(buildRequest());
  const handlePublish = () => {
    if (draftId) publishMutation.mutate({ id: draftId, sendEmail: true });
  };

  const isConfigured = statusQuery.data?.configured ?? false;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Publish</h1>

      {/* Status banner */}
      {statusQuery.isSuccess && !isConfigured && (
        <div className="bg-yellow-900/30 border border-yellow-800 rounded-lg p-3 flex items-center gap-2 text-sm text-yellow-300">
          <AlertTriangle size={16} />
          Substack not configured. Set SUBSTACK_EMAIL, SUBSTACK_PASSWORD, and SUBSTACK_PUBLICATION_URL
          environment variables. You can still compose and preview posts.
        </div>
      )}

      {/* Compose form */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Post Type */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Post Type</label>
            <select
              value={postType}
              onChange={(e) => setPostType(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {POST_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>

          {/* Backtest run selector */}
          {postType === 'backtest_report' && (
            <div>
              <label className="block text-sm text-gray-400 mb-1">Backtest Run</label>
              <select
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a run...</option>
                {runsQuery.data?.runs?.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.strategy} - {run.symbols.join(', ')} ({run.start_date})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Signal ID */}
          {postType === 'signal_note' && (
            <div>
              <label className="block text-sm text-gray-400 mb-1">Signal ID</label>
              <input
                type="text"
                value={signalId}
                onChange={(e) => setSignalId(e.target.value)}
                placeholder="Enter signal ID"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-end gap-2">
            <button
              onClick={handleCompose}
              disabled={composeMutation.isPending}
              className="flex-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white font-medium rounded-lg px-4 py-2 text-sm flex items-center justify-center gap-2 transition-colors"
            >
              <Eye size={16} />
              {composeMutation.isPending ? 'Composing...' : 'Preview'}
            </button>
            <button
              onClick={handleCreateDraft}
              disabled={draftMutation.isPending || !isConfigured}
              className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium rounded-lg px-4 py-2 text-sm flex items-center justify-center gap-2 transition-colors"
              title={!isConfigured ? 'Substack not configured' : ''}
            >
              <FileText size={16} />
              {draftMutation.isPending ? 'Creating...' : 'Draft'}
            </button>
          </div>
        </div>

        {/* Custom post fields */}
        {postType === 'custom' && (
          <div className="mt-4 space-y-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Title</label>
              <input
                type="text"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                placeholder="Post title"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Content (Markdown)</label>
              <textarea
                value={customMarkdown}
                onChange={(e) => setCustomMarkdown(e.target.value)}
                rows={8}
                placeholder="Write your post in markdown..."
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Tags (comma-separated)</label>
              <input
                type="text"
                value={customTags}
                onChange={(e) => setCustomTags(e.target.value)}
                placeholder="trading, ai, signals"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        )}

        {/* Error messages */}
        {composeMutation.isError && (
          <div className="mt-4 bg-red-900/30 border border-red-800 rounded-lg p-3 flex items-center gap-2 text-sm text-red-300">
            <AlertTriangle size={16} />
            {(composeMutation.error as Error)?.message || 'Compose failed'}
          </div>
        )}
        {draftMutation.isError && (
          <div className="mt-4 bg-red-900/30 border border-red-800 rounded-lg p-3 flex items-center gap-2 text-sm text-red-300">
            <AlertTriangle size={16} />
            {(draftMutation.error as Error)?.message || 'Draft creation failed'}
          </div>
        )}
      </div>

      {/* Publish button (shown when draft exists) */}
      {draftId && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Draft #{draftId} ready to publish</p>
            <p className="text-xs text-gray-500 mt-1">This will publish to your Substack and send to subscribers.</p>
          </div>
          <button
            onClick={handlePublish}
            disabled={publishMutation.isPending}
            className="bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium rounded-lg px-6 py-2 text-sm flex items-center gap-2 transition-colors"
          >
            <Send size={16} />
            {publishMutation.isPending ? 'Publishing...' : 'Publish'}
          </button>
        </div>
      )}

      {publishMutation.isSuccess && (
        <div className="bg-green-900/30 border border-green-800 rounded-lg p-3 text-sm text-green-300">
          Post published successfully!
        </div>
      )}

      {/* Preview panel */}
      {preview && <PostPreview post={preview} />}

      {/* Past composed drafts */}
      {composedDrafts.length > 0 && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FileText size={18} />
            Composed Drafts
          </h2>
          <DraftsList
            drafts={composedDrafts}
            onSelect={(draft) => {
              setPreview(draft);
              setSelectedDraftIndex(composedDrafts.indexOf(draft));
            }}
            selectedIndex={selectedDraftIndex}
          />
        </div>
      )}
    </div>
  );
}
