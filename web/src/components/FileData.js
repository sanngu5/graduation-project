import React from 'react';
import VideocamOffIcon from '@material-ui/icons/VideocamOff';
import VideocamIcon from '@material-ui/icons/Videocam';
import CheckCircleIcon from '@material-ui/icons/CheckCircle';
import AutorenewIcon from '@material-ui/icons/Autorenew';

const FileData = (props) => {    
    const {selectedFile, videoStatus} = props

    const styleFileData = {
        "height": 250,
        "width": 400,
        "margin": 20,
        "border": "solid",
        "borderRadius": 15,
        "borderColor": "#E2E2E2",
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "center",
    }

    if(videoStatus==="UPLOADABLE") {
        return(
            <div style={styleFileData}>
                <VideocamIcon fontSize='large'/>
                <h4>업로드 파일 선택 완료</h4>
            </div>
        )
    } else if(videoStatus==="LOADING") {
        return(
            <div style={styleFileData}>
                <AutorenewIcon fontSize='large'/>
                <h4>파일 변환중</h4> 
            </div>
        )
    } else if(videoStatus==="DOWNLOADABLE") {
        return(
            <div style={styleFileData}>
                <CheckCircleIcon fontSize='large' />
                <h4>파일 변환 완료</h4> 
            </div>
        )
    } else {
        return (
            <div style={styleFileData}>
                <VideocamOffIcon fontSize='large' />
                <h4>업로드 파일을 선택하세요</h4>
            </div>
        )
    }
}

export default FileData;