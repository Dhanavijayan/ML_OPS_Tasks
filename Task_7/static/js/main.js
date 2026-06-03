document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('.prediction-form');
  if (!form) return;

  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    if (!button) return;
    button.textContent = 'Predicting...';
    button.disabled = true;
  });
});
