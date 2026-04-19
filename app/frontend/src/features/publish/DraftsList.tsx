import { SubstackPostContent } from '../../types/substack';
import { FileText, Clock } from 'lucide-react';

interface DraftsListProps {
  drafts: SubstackPostContent[];
  onSelect: (draft: SubstackPostContent) => void;
  selectedIndex: number | null;
}

export default function DraftsList({ drafts, onSelect, selectedIndex }: DraftsListProps) {
  if (drafts.length === 0) {
    return (
      <div className="text-gray-500 text-sm">
        No drafts composed yet. Use the form above to create one.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {drafts.map((draft, idx) => (
        <button
          key={idx}
          onClick={() => onSelect(draft)}
          className={`w-full text-left bg-gray-800 hover:bg-gray-700 rounded-lg p-3 transition-colors ${
            selectedIndex === idx ? 'ring-1 ring-blue-500' : ''
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText size={16} className="text-blue-400" />
              <span className="font-medium text-sm truncate">{draft.title}</span>
            </div>
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Clock size={12} />
              {new Date(draft.created_at).toLocaleTimeString()}
            </div>
          </div>
          {draft.subtitle && (
            <p className="text-xs text-gray-500 mt-1 truncate ml-7">{draft.subtitle}</p>
          )}
        </button>
      ))}
    </div>
  );
}
