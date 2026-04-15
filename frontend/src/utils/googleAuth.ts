// Google OAuth handler
export const handleGoogleLogin = async (credentialResponse: any, onSuccess: (data: any) => void, onError: (err: string) => void) => {
  try {
    // The credential is a JWT token from Google
    const token = credentialResponse.credential;
    
    // Send the token to your backend
    const response = await fetch(`${import.meta.env.VITE_API_URL || 'https://mbaza-mzf3.onrender.com'}/api/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });

    if (!response.ok) {
      const error = await response.json();
      onError(error.detail || 'Google login failed');
      return;
    }

    const data = await response.json();
    
    // Store auth data
    localStorage.setItem('auth_token', data.access_token);
    localStorage.setItem('user_data', JSON.stringify(data.user));
    
    // Trigger success callback
    onSuccess(data);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : 'Google login failed';
    console.error('Google login error:', errorMsg);
    onError(errorMsg);
  }
};
