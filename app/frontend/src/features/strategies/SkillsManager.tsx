import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ToggleLeft, ToggleRight, Tag } from 'lucide-react';
import { fetchSkills, toggleSkill } from '../../lib/api';
import { AgentType } from '../../types/agent';

export default function SkillsManager() {
  const queryClient = useQueryClient();
  const [agentType, setAgentType] = useState<AgentType>('swing');

  const { data, isLoading } = useQuery({
    queryKey: ['skills', agentType],
    queryFn: () => fetchSkills(agentType),
    refetchInterval: 10000,
  });

  const toggleMutation = useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      toggleSkill({ agentType, name, enabled }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['skills', agentType] }),
  });

  const skills = data?.skills || [];

  if (isLoading) {
    return (
      <div className="p-6 flex justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white" />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">Strategies</h2>
        <span className="text-sm text-gray-500">
          {skills.filter((s) => s.enabled).length} of {skills.length} enabled
        </span>
      </div>

      {/* Agent type tabs */}
      <div className="flex gap-2">
        {(['swing', 'day'] as AgentType[]).map((type) => (
          <button
            key={type}
            onClick={() => setAgentType(type)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              agentType === type
                ? 'bg-blue-600/20 text-blue-400 border border-blue-500/50'
                : 'bg-gray-900 text-gray-400 border border-gray-800 hover:border-gray-700'
            }`}
          >
            {type === 'swing' ? 'Swing Agent' : 'Day Agent'}
          </button>
        ))}
      </div>

      <div className="grid gap-4">
        {skills.map((skill) => (
          <div
            key={skill.name}
            className={`bg-gray-900 rounded-lg border p-5 transition-colors ${
              skill.enabled ? 'border-gray-800' : 'border-gray-800/50 opacity-60'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <h3 className="font-semibold text-lg">{skill.name.replace(/_/g, ' ')}</h3>
                <div className="flex gap-1.5">
                  {skill.tags.map((tag) => (
                    <span
                      key={tag}
                      className="flex items-center gap-1 text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded"
                    >
                      <Tag size={10} />
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              <button
                onClick={() =>
                  toggleMutation.mutate({ name: skill.name, enabled: !skill.enabled })
                }
                className="text-2xl transition-colors"
                title={skill.enabled ? 'Disable skill' : 'Enable skill'}
              >
                {skill.enabled ? (
                  <ToggleRight className="text-green-400" size={32} />
                ) : (
                  <ToggleLeft className="text-gray-600" size={32} />
                )}
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Timeframes</span>
                <div className="mt-1 font-medium">{skill.timeframes.join(', ')}</div>
              </div>
              <div>
                <span className="text-gray-500">Risk per Trade</span>
                <div className="mt-1 font-medium">{(skill.risk_per_trade * 100).toFixed(1)}%</div>
              </div>
              <div>
                <span className="text-gray-500">Max Positions</span>
                <div className="mt-1 font-medium">{skill.max_positions}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {skills.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No strategies loaded</p>
          <p className="text-gray-600 text-sm mt-2">
            Add .md skill files to the app/agent/skills/specs/{agentType}/ directory
          </p>
        </div>
      )}
    </div>
  );
}
