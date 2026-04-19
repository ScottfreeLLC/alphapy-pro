import { SubstackPostContent } from '../../types/substack';

interface PostPreviewProps {
  post: SubstackPostContent;
}

export default function PostPreview({ post }: PostPreviewProps) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 space-y-4">
      <div>
        <h2 className="text-xl font-bold">{post.title}</h2>
        {post.subtitle && (
          <p className="text-sm text-gray-400 mt-1">{post.subtitle}</p>
        )}
      </div>

      {post.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {post.tags.map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 bg-gray-800 text-gray-400 rounded-full"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="border-t border-gray-800 pt-4">
        {post.sections.map((section, idx) => (
          <div key={idx} className="mb-4">
            {section.heading && (
              <h3 className="text-lg font-semibold mb-2">{section.heading}</h3>
            )}
            <div
              className="text-sm text-gray-300 prose prose-invert prose-sm max-w-none [&_table]:w-full [&_table]:text-sm [&_td]:py-1 [&_td]:px-2 [&_th]:py-1 [&_th]:px-2 [&_th]:text-left [&_th]:text-gray-400 [&_tr]:border-b [&_tr]:border-gray-800"
              dangerouslySetInnerHTML={{ __html: section.content }}
            />
          </div>
        ))}
      </div>

      <div className="text-xs text-gray-600 pt-2 border-t border-gray-800">
        Created: {new Date(post.created_at).toLocaleString()}
      </div>
    </div>
  );
}
