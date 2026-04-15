import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Heart, Mail, Lock, User, ArrowRight } from 'lucide-react';

interface RegisterProps {
  onSwitchToLogin: () => void;
}

export const Register: React.FC<RegisterProps> = ({ onSwitchToLogin }) => {
  const { register } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    console.log('Register attempt with:', { email, name, password });

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    try {
      await register(email, password);
      console.log('Registration successful!');
      setSuccess('Account created successfully. Please sign in.');
      onSwitchToLogin();
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Registration failed. Please try again.';
      console.error('Registration error:', errorMsg);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0c10] to-[#161b22] flex items-center justify-center p-4 md:p-8">
      <div className="w-full max-w-md">
        <div className="bg-[#0d1117] border border-[#30363d] rounded-3xl p-8 shadow-2xl">
          {/* Header */}
          <div className="mb-8 text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <Heart className="text-[#ff7eb3]" size={32} />
              <h1 className="text-3xl font-black text-white">Mbaza</h1>
            </div>
            <p className="text-sm text-[#6b7fa8] font-medium">Create Your Account</p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email */}
            <div>
              <label className="block text-xs font-bold text-[#c9d1d9] uppercase tracking-[0.1em] mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-3.5 text-[#6b7fa8]" size={18} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full bg-[#161b22] border border-[#30363d] rounded-xl pl-12 pr-4 py-3 text-white placeholder-[#6b7fa8] focus:outline-none focus:border-[#29d8a8] transition-colors"
                  required
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-bold text-[#c9d1d9] uppercase tracking-[0.1em] mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-3.5 text-[#6b7fa8]" size={18} />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="At least 6 characters"
                  className="w-full bg-[#161b22] border border-[#30363d] rounded-xl pl-12 pr-4 py-3 text-white placeholder-[#6b7fa8] focus:outline-none focus:border-[#29d8a8] transition-colors"
                  required
                />
              </div>
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-xs font-bold text-[#c9d1d9] uppercase tracking-[0.1em] mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-3.5 text-[#6b7fa8]" size={18} />
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm password"
                  className="w-full bg-[#161b22] border border-[#30363d] rounded-xl pl-12 pr-4 py-3 text-white placeholder-[#6b7fa8] focus:outline-none focus:border-[#29d8a8] transition-colors"
                  required
                />
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                <p className="text-xs text-red-400 font-medium">{error}</p>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                <p className="text-xs text-green-400 font-medium">{success}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-[#ff7eb3] to-[#ff6fa0] text-white font-black py-3 rounded-xl hover:shadow-lg hover:shadow-[#ff7eb3]/20 transition-all disabled:opacity-50 uppercase text-sm tracking-wider flex items-center justify-center gap-2"
            >
              {loading ? 'Creating Account...' : (
                <>
                  Sign Up <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>

          {/* Login Link */}
          <p className="text-center mt-6 text-sm text-[#6b7fa8]">
            Already have an account?{' '}
            <button
              type="button"
              onClick={onSwitchToLogin}
              className="text-[#29d8a8] font-bold hover:underline"
            >
              Sign In
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};
