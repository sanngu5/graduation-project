import React from 'react';
import STATUS from '../constants/Status';

const FileData = (props) => {    
    const {selectedFile, videoStatus} = props

    if(selectedFile) {
        return(
            <div>
                <br/>
                <h4>업로드 파일 선택 완료</h4>
                <p>파일명: {selectedFile.name}</p>
                <p>파일유형: {selectedFile.type}</p>
            </div>
        )
    } else if(videoStatus===STATUS["LOADING"]) {
        return(
            <div>
                <br/>
                <h4>파일 변환중</h4> 
            </div>
        )
    } else if(videoStatus===STATUS["DONWLOADABLE"]) {
        return(
            <div>
                <br/>
                <h4>파일 변환 완료</h4> 
            </div>
        )
    } else {
        return (
            <div>
                <br/>
                <h4>업로드 파일을 선택하세요</h4>
            </div>
        )
    }
}

export default FileData;