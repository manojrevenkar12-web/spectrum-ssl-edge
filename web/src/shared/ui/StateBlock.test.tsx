import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { EmptyState, ErrorState, Skeleton } from './StateBlock';

describe('state blocks', () => {
  it('an error explains itself and retries', () => {
    const retry = vi.fn();
    render(<ErrorState message="The results file could not be loaded." onRetry={retry} />);
    expect(screen.getByRole('alert').textContent).toContain('could not be loaded');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(retry).toHaveBeenCalledOnce();
  });

  it('a labelled skeleton is announced, an unlabelled one is hidden', () => {
    const { container } = render(
      <>
        <Skeleton height={10} label="Loading results" />
        <Skeleton height={10} />
      </>,
    );
    expect(screen.getByRole('status', { name: 'Loading results' })).toBeTruthy();
    expect(container.querySelectorAll('[aria-hidden="true"]').length).toBe(1);
  });

  it('an empty state says what to do', () => {
    render(<EmptyState title="No results yet">Run sse all.</EmptyState>);
    expect(screen.getByText('No results yet')).toBeTruthy();
  });
});
