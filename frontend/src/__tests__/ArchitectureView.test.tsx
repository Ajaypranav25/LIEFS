import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ArchitectureView } from '../components/ArchitectureView';

describe('ArchitectureView Component', () => {
  it('renders all 5 architectural stage cards', () => {
    render(<ArchitectureView />);
    expect(screen.getByText(/Stage 1: Naive Baseline/i)).toBeInTheDocument();
    expect(screen.getByText(/Stage 2: KV-Cache/i)).toBeInTheDocument();
    expect(screen.getByText(/Stage 3: Cont\. Batching/i)).toBeInTheDocument();
    expect(screen.getByText(/Stage 4: Paged Attention/i)).toBeInTheDocument();
    expect(screen.getByText(/Stage 5: INT8 Quantized/i)).toBeInTheDocument();
  });

  it('renders the interactive KV-cache memory sizing calculator and updates calculations on preset change', () => {
    render(<ArchitectureView />);
    expect(screen.getByText(/Interactive KV-Cache GPU Memory Sizing Calculator/i)).toBeInTheDocument();

    const llamaPresetBtn = screen.getByText(/Llama-3-8B \(GQA\)/i);
    fireEvent.click(llamaPresetBtn);

    // After switching to LLaMA 3, 32 layers and 8 KV heads should be active
    expect(screen.getByText('32')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
  });
});
