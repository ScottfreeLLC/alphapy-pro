export interface SubstackPostSection {
  heading: string;
  content: string;
  type: string;
}

export interface SubstackPostContent {
  title: string;
  subtitle: string;
  sections: SubstackPostSection[];
  tags: string[];
  html: string;
  created_at: string;
}

export interface SubstackDraft {
  id: number;
  slug?: string;
  draft_title?: string;
  draft_subtitle?: string;
  audience?: string;
}

export interface SubstackStatus {
  configured: boolean;
  publication_url: string | null;
}

export interface SubstackComposeRequest {
  post_type: 'daily_signals' | 'backtest_report' | 'signal_note' | 'custom';
  run_id?: string;
  signal_id?: string;
  title?: string;
  markdown?: string;
  tags?: string[];
}
