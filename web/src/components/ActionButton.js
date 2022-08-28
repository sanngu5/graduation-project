import React from 'react';
import STATUS from '../constants/Status';

const ActionButton = (props) => {
    const {
        selectedFile,
        videoStatus,
        onFileUpload,
        onFileChanged,
        onFileDownload
    } = props;

    if(videoStatus === STATUS["UPLOADABLE"]) {
        return (
            <button disabled={!selectedFile} onClick={onFileUpload}>
                동영상 변환
            </button>
        )
    } else if (videoStatus === STATUS["LOADING"]) {
        return(
            <button disabled={true} onClick={onFileChanged}>
                변환중
            </button>
        )
    } else if (videoStatus === STATUS["DOWNLOADABLE"]) {
        return (
            // <a href="http://127.0.0.1:5000/download">
            //     다운로드
            // </a>
            <button 
                // onClick={onFileDownload}
                href="http://127.0.0.1:5000/download"
                
            >
                다운로드
            </button>
        )
    } else {
        return (<></>)
    }
}

export default ActionButton;