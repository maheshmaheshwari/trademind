import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

export default function Layout() {
    return (
        <div className="flex flex-col min-h-screen bg-background-dark">
            <Navbar />
            <main className="flex flex-1 justify-center py-8 px-4 md:px-10">
                <div className="flex flex-col w-full max-w-[1200px] flex-1 gap-6">
                    <Outlet />
                </div>
            </main>
        </div>
    );
}
