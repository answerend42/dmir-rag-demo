/**
 * @file RandomImage.jsx
 * @brief 预览面板的空状态插图组件。
 */
import React, { useState, useEffect } from 'react';
import image1 from '../assets/01.jpg';
import image2 from '../assets/02.jpg';
import image3 from '../assets/03.jpg';
import image4 from '../assets/04.jpg';
import image5 from '../assets/05.jpg';
import image6 from '../assets/06.jpg';
import image7 from '../assets/07.jpg';
import image8 from '../assets/08.jpg';
import image9 from '../assets/09.jpg';
import image10 from '../assets/10.jpg';

/**
 * @brief 显示一张随机内置图片和对应的空状态消息。
 * @param {{message: string}} props 组件属性。
 * @returns {JSX.Element} 空状态插图和消息。
 */
const RandomImage = ({ message }) => {
  const [randomImage, setRandomImage] = useState(null);

  useEffect(() => {
    const images = [
      image1, image2, image3, image4, image5,
      image6, image7, image8, image9, image10
    ];
    const randomIdx = Math.floor(Math.random() * images.length);
    setRandomImage(images[randomIdx]);
  }, []);

  return (
    <div className="flex h-full min-h-[420px] flex-col items-center justify-center p-6 text-gray-500">
      {randomImage && (
        <div className="w-full max-w-2xl overflow-hidden rounded border bg-white p-2 shadow-sm">
          <img
            src={randomImage}
            alt="空状态插图"
            className="h-[260px] w-full rounded object-cover"
          />
        </div>
      )}
      <p className="mt-5 max-w-xl text-center text-sm leading-7">{message}</p>
    </div>
  );
};

export default RandomImage;
