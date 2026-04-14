import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Heart, Mail, Lock, ArrowRight } from 'lucide-react';

interface LoginProps {
  onSwitchToRegister: () => void;
}

export const Login: React.FC<LoginProps> = ({ onSwitchToRegister }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    console.log('Login attempt with:', { email, password });

    try {
      await login(email, password);
      console.log('Login successful!');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Login failed. Please try again.';
      console.error('Login error:', errorMsg);
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
            <p className="text-sm text-[#6b7fa8] font-medium">Sexual Health Education & Period Tracker</p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
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
                  placeholder="••••••••"
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

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-[#29d8a8] to-[#20a88f] text-[#0a0c10] font-black py-3 rounded-xl hover:shadow-lg hover:shadow-[#29d8a8]/20 transition-all disabled:opacity-50 uppercase text-sm tracking-wider flex items-center justify-center gap-2"
            >
              {loading ? 'Logging in...' : (
                <>
                  Sign In <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="my-6 flex items-center gap-3">
            <div className="flex-1 border-t border-[#30363d]"></div>
            <span className="text-xs text-[#6b7fa8] font-bold uppercase">OR</span>
            <div className="flex-1 border-t border-[#30363d]"></div>
          </div>

          {/* Google Login (Coming Soon) */}
          <button
            type="button"
            disabled
            className="w-full border border-[#30363d] bg-[#161b22]/50 text-[#6b7fa8] font-bold py-3 rounded-xl opacity-50 cursor-not-allowed uppercase text-sm tracking-wider"
            title="Google login coming soon"
          >
            Continue with Google (Coming Soon)
          </button>

          {/* Register Link */}
          <p className="text-center mt-6 text-sm text-[#6b7fa8]">
            Don't have an account?{' '}
            <button
              type="button"
              onClick={onSwitchToRegister}
              className="text-[#29d8a8] font-bold hover:underline"
            >
              Sign Up
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};
