interface LoadingProps {
    isLoading: boolean;
    text?: string;
}

const Loading = ({ isLoading, text = 'Preparing results...' }: LoadingProps) => {
    if (!isLoading) return null;

    return (
        <div className="flex flex-col justify-center items-center h-full w-full min-h-[200px]">
            <div className="w-9 h-9 border-4 border-white/30 border-l-[#15a1ff] rounded-full animate-spin"></div>
            <p className="mt-2.5 text-base text-[#eeeeee]">{text}</p>
        </div>
    );
};

export default Loading;
